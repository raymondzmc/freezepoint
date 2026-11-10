"""Tests for the three arms. No data: `validate` runs for real with its dry run pointed at an empty directory, and
`evaluate` is a fake that scores by expression length (feedback ic = len/100, pilot ic = -len/100 by default)."""
import copy
import json
import math
import pathlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from harness import backtest
from harness.backtest import FactorScore
from harness.ledger.ledger import Ledger, LoopView, Trial
from harness.pit import tools
from harness.splits import get as get_split
from loop import config
from loop.arms import run_arm
from loop.generator import FakeGenerator

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = config.load(ROOT / "configs" / "cn_refine.yaml")
SPLIT = get_split(BASE.split)
BAD = "Ref($close,-1)"     # rejected by the validator: reads the future


@pytest.fixture(autouse=True)
def no_data_dir(monkeypatch, tmp_path):
    """Point the validator's dry run at an empty directory so these tests never touch Qlib data."""
    monkeypatch.setattr(tools, "QLIB_DATA", tmp_path / "no-qlib-data")


def cfg_for(arm, k=3, **kw):
    return replace(BASE, arm=arm, budget=replace(BASE.budget, max_steps=k, n_candidates=k + 1), **kw)


def fake_evaluate(fail=(), pilot=lambda ic: -ic):
    """Score by length. `fail` lists expressions whose backtest fails (ok False); `pilot` maps the feedback ic to the
    pilot ic, so a leak of pilot into a decision changes the outcome."""
    calls = []
    def ev(expr, window, universe, region, quantile=0.2):
        calls.append((expr, window, universe, region))
        if expr in fail:
            nan = float("nan")
            return FactorScore(False, f"no IC for {expr}", nan, nan, nan, nan, nan, 0, [], [])
        ic, turnover = len(expr) / 100, 0.5
        if window == SPLIT.pilot: ic, turnover = pilot(ic), 0.7
        return FactorScore(True, "", ic, 2 * ic, ic / 10, 3 * ic, turnover, 3, [ic, -ic, ic],
                           ["2018-01-02", "2018-01-03", "2018-01-04"])
    ev.calls = calls
    return ev


class SpyGenerator:
    """Wraps FakeGenerator, records every call (with deep copies of what it was handed), sets a cost, and can raise
    on the call with index `raise_at`."""
    def __init__(self, exprs, raise_at=None):
        self.inner, self.calls, self.raise_at, self.last_cost = FakeGenerator(exprs), [], raise_at, {}
    def _start(self, call):
        if self.raise_at is not None and len(self.calls) == self.raise_at: raise RuntimeError("API outage")
        self.calls.append(call)
        self.last_cost = {"input_tokens": len(self.calls), "seconds": 0.25, "model": "served-model"}
    def propose(self, context, n, seed):
        self._start({"kind": "propose", "context": context, "n": n, "seed": seed})
        return self.inner.propose(context, n, seed)
    def refine(self, expr, feedback, history, seed, *, context=""):
        self._start({"kind": "refine", "expr": expr, "feedback": copy.deepcopy(feedback),
                     "history": copy.deepcopy(history), "seed": seed, "context": context})
        return self.inner.refine(expr, feedback, history, seed, context=context)


def run(arm, exprs, tmp_path, k=3, seed=0, ev=None, gen=None, **kw):
    cfg = cfg_for(arm, k)
    gen = gen or SpyGenerator(exprs)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    chosen = run_arm(cfg, seed, gen, ledger, ev or fake_evaluate(), root=tmp_path / "runs", **kw)
    rid = config.run_id(cfg, seed)
    trials = sorted((t for t in ledger.trials().values() if t["run"] == rid), key=lambda t: t["step"])
    return SimpleNamespace(cfg=cfg, rid=rid, gen=gen, ledger=ledger, chosen=chosen, trials=trials)


def chosen_json(tmp_path, rid):
    return json.loads((tmp_path / "runs" / rid / "chosen.json").read_text())


# the brief's four tests
def test_refine_logs_every_candidate_and_keeps_best(tmp_path):
    exprs = ["$close", "Mean($close, 5) / $close", "$open", "Mean($close, 20)"]
    r = run("refine", exprs, tmp_path)
    assert r.ledger.count(r.rid) == r.cfg.budget.max_steps + 1
    assert r.chosen == r.trials[1]["id"]                               # the longest expression has the highest ic
    assert [t["step"] for t in r.trials] == [0, 1, 2, 3]
    assert [t["parent"] for t in r.trials] == [None, r.trials[0]["id"], r.trials[1]["id"], r.trials[1]["id"]]
    assert chosen_json(tmp_path, r.rid) == {"id": r.chosen, "artifact": exprs[1], "arm": "refine", "run": r.rid,
                                            "seed": 0, "step": 1}

def test_bestofn_uses_equal_generations(tmp_path):
    r = run("bestofn", ["$close", "Mean($close, 5)", "$open"], tmp_path)
    assert r.ledger.count(r.rid) == r.cfg.budget.n_candidates
    assert {t["arm"] for t in r.trials} == {"bestofn"} and all(t["parent"] is None for t in r.trials)

def test_invalid_candidate_is_logged_and_loop_continues(tmp_path):
    r = run("refine", ["$close", BAD, "Mean($close, 5)", "$open"], tmp_path)
    bad = [t for t in r.trials if not t["valid"]]
    assert len(bad) == 1 and bad[0]["artifact"] == BAD and "future" in bad[0]["error"] and bad[0]["scores"] == {}
    assert r.chosen == r.trials[2]["id"] and len(r.trials) == 4
    assert r.trials[2]["parent"] == r.trials[0]["id"]                  # refined from the best, not from the invalid one

def test_loop_never_reads_pilot(tmp_path, monkeypatch):
    seen, hides, depth = [], [], [0]
    real_view, real_trials, real_records = LoopView.trials, Ledger.trials, Ledger.records
    def view_spy(self):
        out = real_view(self); seen.append(copy.deepcopy(out)); return out
    def trials_spy(self, hide=()):
        hides.append(tuple(hide)); depth[0] += 1
        try: return real_trials(self, hide)
        finally: depth[0] -= 1
    def records_spy(self):
        assert depth[0] > 0, "the ledger's raw records were read outside Ledger.trials"
        return real_records(self)
    monkeypatch.setattr(LoopView, "trials", view_spy)
    monkeypatch.setattr(Ledger, "trials", trials_spy)
    monkeypatch.setattr(Ledger, "records", records_spy)
    for arm in ("frozen", "refine", "bestofn"):
        run_arm(cfg_for(arm), 0, SpyGenerator(["$close", "Mean($close, 5)", BAD, "$open"]),
                Ledger(tmp_path / arm / "ledger.jsonl"), fake_evaluate(), root=tmp_path / "runs")
    assert seen and all("pilot" not in t["scores"] for out in seen for t in out.values())
    assert hides and all("pilot" in h for h in hides)              # every read went through the loop's view
    monkeypatch.undo()
    raw = Ledger(tmp_path / "refine" / "ledger.jsonl").trials()
    assert all("pilot" in t["scores"] for t in raw.values() if t["valid"])   # pilot was written all the same


# frozen and best-of-N
def test_frozen_is_one_generation_one_trial_chosen(tmp_path):
    r = run("frozen", ["Mean($close, 5)", "$close"], tmp_path, seed=3,
                                                tool_log_span=(4, 9))
    assert len(r.gen.calls) == 1 and r.gen.calls[0] == {"kind": "propose", "context": "", "n": 1, "seed": 3}
    assert len(r.trials) == 1 and r.chosen == r.trials[0]["id"] and r.trials[0]["step"] == 0
    t = r.trials[0]
    assert t["cost"] == {"input_tokens": 1, "seconds": 0.25, "model": "served-model"} and t["model"] == "served-model"
    assert t["tool_log_span"] == [4, 9] and t["arm"] == "frozen" and t["run"] == r.rid
    assert chosen_json(tmp_path, r.rid)["id"] == r.chosen

def test_model_falls_back_to_config_when_cost_has_none(tmp_path):
    r = run("frozen", ["$close"], tmp_path, gen=FakeGenerator(["$close"]))
    assert r.trials[0]["model"] == r.cfg.generator.model and r.trials[0]["cost"] == {}

def test_context_is_handed_to_propose(tmp_path):
    r = run("bestofn", ["$close"], tmp_path, context="csi300 up to 2021-12-31")
    assert {c["context"] for c in r.gen.calls} == {"csi300 up to 2021-12-31"}

def test_context_is_handed_to_every_refinement(tmp_path):
    r = run("refine", ["$close", BAD, "Mean($close, 5)", "$open"], tmp_path, context="csi300 up to 2021-12-31")
    kinds = [c["kind"] for c in r.gen.calls]
    assert kinds.count("refine") == 3 and {c["context"] for c in r.gen.calls} == {"csi300 up to 2021-12-31"}

def test_bestofn_repeated_expressions_get_distinct_ids_steps_and_seeds(tmp_path):
    r = run("bestofn", ["$close"], tmp_path, k=4, seed=7)
    n = r.cfg.budget.n_candidates
    assert len({t["id"] for t in r.trials}) == n and len(r.trials) == n
    assert [t["step"] for t in r.trials] == list(range(n))
    assert [c["seed"] for c in r.gen.calls] == [7 + i for i in range(n)]
    assert {c["kind"] for c in r.gen.calls} == {"propose"} and all(c["n"] == 1 for c in r.gen.calls)
    assert r.chosen == r.trials[0]["id"]                               # all tie: the first is kept

@pytest.mark.parametrize("arm, budget", [("frozen", lambda b: 1), ("refine", lambda b: b.max_steps + 1),
                                         ("bestofn", lambda b: b.n_candidates)])
def test_generator_calls_equal_the_budget(tmp_path, arm, budget):
    r = run(arm, ["$close", BAD, "Mean($close, 5)", "$open", "Rank($close, 10)"],
                                                tmp_path, k=5)
    assert len(r.gen.calls) == budget(r.cfg.budget) == len(r.trials)

@pytest.mark.parametrize("arm", ["frozen", "refine", "bestofn"])
def test_every_append_precedes_its_scores(tmp_path, arm):
    r = run(arm, ["$close", BAD, "Mean($close, 5)", "$open"], tmp_path)
    recs = r.ledger.records()
    appended = {r["id"]: i for i, r in enumerate(recs) if "artifact" in r}
    scores = [(i, r) for i, r in enumerate(recs) if "score" in r]
    assert scores and all(r["id"] in appended and appended[r["id"]] < i for i, r in scores)
    for t in r.trials:   # a valid candidate gets exactly one feedback and one pilot record, an invalid one none
        windows = sorted(w for _, r in scores if r["id"] == t["id"] for w in r["score"])
        assert windows == (["feedback", "pilot"] if t["valid"] else [])

def test_score_records_carry_the_series(tmp_path):
    r = run("frozen", ["$close"], tmp_path)
    fb = r.trials[0]["scores"]["feedback"]
    assert fb["ic"] == 0.06 and fb["daily_ls"] == [0.06, -0.06, 0.06] and fb["dates"][0] == "2018-01-02"
    assert fb["ok"] is True and fb["n_days"] == 3

@pytest.mark.parametrize("arm", ["refine", "bestofn"])
def test_choice_follows_feedback_not_pilot(tmp_path, arm):
    exprs = ["$close", "Mean($close, 5) / $close", "$open", "Mean($close, 5)"]
    r = run(arm, exprs, tmp_path)
    raw = r.ledger.trials()
    assert raw[r.chosen]["artifact"] == "Mean($close, 5) / $close"
    best_pilot = max((t for t in raw.values() if t["valid"]), key=lambda t: t["scores"]["pilot"]["ic"])
    assert best_pilot["id"] != r.chosen                              # pilot would have chosen another candidate

def test_everything_the_generator_receives_comes_from_the_loop_view(tmp_path):
    exprs = ["$close", BAD, "Mean($close, 5) / $close", "$open", "Mean($close, 5)", "Rank($close, 10)"]
    r = run("refine", exprs, tmp_path, k=5,
                                                ev=fake_evaluate(fail=("$open",), pilot=lambda ic: ic + 0.5))
    view = {t["id"]: t for t in LoopView(r.ledger).trials().values()}
    fb_scores = [t["scores"]["feedback"] for t in view.values() if t["scores"].get("feedback", {}).get("ok")]
    refines = [c for c in r.gen.calls if c["kind"] == "refine"]
    assert refines
    for c in refines:
        assert any(all(c["feedback"][k] == s[k] for k in ("ic", "icir", "ls_sharpe", "turnover")) for s in fb_scores)
        assert set(c["feedback"]) == {"ic", "icir", "ls_sharpe", "turnover"}
        for h in c["history"]:
            match = [t for t in view.values() if t["artifact"] == h["expr"]]
            assert match
            if "ic" in h:
                assert any(t["scores"]["feedback"]["ic"] == h["ic"] for t in match)
            else:
                assert any(h["error"] in (t["error"], t["scores"].get("feedback", {}).get("reason")) for t in match)
    # history grows by one entry per candidate, in order
    assert [len(c["history"]) for c in r.gen.calls if c["kind"] == "refine"] == [
        i for i, c in enumerate(r.gen.calls) if c["kind"] == "refine"]


# refine edge cases
def test_refine_with_invalid_first_proposal_proposes_again(tmp_path):
    r = run("refine", [BAD, "$close", "Mean($close, 5)"], tmp_path, k=2, seed=0)
    assert [c["kind"] for c in r.gen.calls] == ["propose", "propose", "refine"]
    assert [c["seed"] for c in r.gen.calls] == [0, 0, 0]
    assert [t["step"] for t in r.trials] == [0, 1, 2]
    assert [t["parent"] for t in r.trials] == [None, None, r.trials[1]["id"]]
    assert not r.trials[0]["valid"] and r.chosen == r.trials[2]["id"]
    assert r.gen.calls[2]["history"][0]["expr"] == BAD and "future" in r.gen.calls[2]["history"][0]["error"]

def test_refine_failed_backtest_is_recorded_never_chosen_and_reported_as_error(tmp_path):
    long_fail = "Mean($close, 5) / $close - 1 + Std($close, 20)"
    ev = fake_evaluate(fail=(long_fail,))
    r = run("refine", ["$close", long_fail, "$open", "$high"], tmp_path, ev=ev)
    failed = r.trials[1]
    assert failed["valid"] and failed["scores"]["feedback"]["ok"] is False
    assert failed["scores"]["feedback"]["ic"] is None and "no IC" in failed["scores"]["feedback"]["reason"]
    assert any(e == long_fail and w == SPLIT.pilot for e, w, _, _ in ev.calls)   # pilot still attempted
    assert r.chosen != failed["id"] and r.chosen == r.trials[0]["id"]
    assert r.trials[2]["parent"] == r.trials[0]["id"]
    assert r.gen.calls[2]["history"][1] == {"expr": long_fail, "error": f"no IC for {long_fail}"}
    assert r.gen.calls[2]["history"][0] == {"expr": "$close", "ic": 0.06}

def test_refine_keeps_the_earlier_of_two_equal_candidates(tmp_path):
    r = run("refine", ["Mean($close, 5)", "Mean($close, 6)", "$open"], tmp_path, k=2)
    assert r.chosen == r.trials[0]["id"] and r.trials[2]["parent"] == r.trials[0]["id"]
    assert r.gen.calls[2]["expr"] == "Mean($close, 5)"

def test_refine_feedback_is_the_best_candidates(tmp_path):
    r = run("refine", ["$close", "$open"], tmp_path, k=1)
    assert r.gen.calls[1]["feedback"] == {"ic": 0.06, "icir": 2 * 0.06, "ls_sharpe": 3 * 0.06, "turnover": 0.5}
    assert r.gen.calls[1]["expr"] == "$close" and r.gen.calls[1]["seed"] == 0


# outcomes and failures
@pytest.mark.parametrize("arm", ["frozen", "refine", "bestofn"])
def test_all_invalid_writes_null_choice_and_returns_none(tmp_path, arm):
    r = run(arm, [BAD, "Magic($close)"], tmp_path, seed=2)
    assert r.chosen is None and r.trials and not any(t["valid"] for t in r.trials)
    assert chosen_json(tmp_path, r.rid) == {"id": None, "artifact": None, "arm": arm, "run": r.rid, "seed": 2,
                                            "step": None}

def test_all_failed_backtests_choose_nothing(tmp_path):
    ev = fake_evaluate(fail=("$close", "$open"))
    r = run("bestofn", ["$close", "$open"], tmp_path, ev=ev)
    assert r.chosen is None and chosen_json(tmp_path, r.rid)["id"] is None and len(r.trials) == 4

@pytest.mark.parametrize("arm", ["refine", "bestofn"])
def test_generator_error_propagates_and_writes_no_choice(tmp_path, arm):
    gen = SpyGenerator(["$close", "Mean($close, 5)", "$open"], raise_at=2)
    with pytest.raises(RuntimeError, match="outage"):
        run(arm, None, tmp_path, gen=gen)
    cfg = cfg_for(arm)
    rid = config.run_id(cfg, 0)
    assert not (tmp_path / "runs" / rid / "chosen.json").exists()
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.count(rid) == 2 and all(t["valid"] for t in ledger.trials().values())

def test_stopping_rule_other_than_none_raises(tmp_path):
    cfg = cfg_for("refine", stopping=replace(BASE.stopping, rule="fixed_k"))
    gen = SpyGenerator(["$close"])
    with pytest.raises(ValueError, match="not implemented"):
        run_arm(cfg, 0, gen, Ledger(tmp_path / "l.jsonl"), fake_evaluate(), root=tmp_path / "runs")
    assert gen.calls == [] and not (tmp_path / "l.jsonl").exists()

def test_metric_other_than_ic_raises(tmp_path):
    cfg = cfg_for("refine", verifier=replace(BASE.verifier, metric="ls_sharpe"))
    with pytest.raises(ValueError, match="not implemented"):
        run_arm(cfg, 0, SpyGenerator(["$close"]), Ledger(tmp_path / "l.jsonl"), fake_evaluate(), root=tmp_path / "r")


@pytest.mark.parametrize("arm", ["frozen", "refine", "bestofn"])
def test_a_ledger_that_already_holds_this_runs_trials_is_refused(tmp_path, arm):
    """A second loop appending into the same run would double its trials: run_arm raises before generating."""
    cfg = cfg_for(arm)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(Trial(arm=arm, run=config.run_id(cfg, 0), step=0, artifact="$close"))
    ledger.append(Trial(arm=arm, run="another-run", step=0, artifact="$close"))
    gen = SpyGenerator(["$close"])
    with pytest.raises(RuntimeError, match="already holds"):
        run_arm(cfg, 0, gen, ledger, fake_evaluate(), root=tmp_path / "runs")
    assert gen.calls == [] and ledger.count() == 2
    assert not (tmp_path / "runs" / config.run_id(cfg, 0) / "chosen.json").exists()


def test_trials_of_another_run_in_the_ledger_do_not_block(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(Trial(arm="frozen", run="another-run", step=0, artifact="$close"))
    assert run_arm(cfg_for("frozen"), 0, SpyGenerator(["$close"]), ledger, fake_evaluate(), root=tmp_path / "r")


def test_fake_evaluate_reverses_pilot():
    ev = fake_evaluate()
    assert math.isclose(ev("$close", SPLIT.pilot, "csi300", "cn").ic, -ev("$close", SPLIT.feedback, "csi300", "cn").ic)


def test_missing_data_propagates_out_of_run_arm(tmp_path, monkeypatch):
    """The real evaluate with no data: the candidate is logged (it validated), then the run stops without a choice."""
    monkeypatch.setattr(backtest, "_MEMO", {})
    cfg = cfg_for("frozen")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(FileNotFoundError, match="cn_data"):
        run_arm(cfg, 0, FakeGenerator(["$close"]), ledger, root=tmp_path / "runs")
    assert ledger.count() == 1 and not (tmp_path / "runs" / config.run_id(cfg, 0) / "chosen.json").exists()
    assert all("score" not in r for r in ledger.records())
