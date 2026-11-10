"""Tests for the post-run scorer. No data: `evaluate` is a fake that records what it was asked for."""
import itertools
import pathlib
from dataclasses import replace

import pytest

from harness import backtest
from harness.backtest import FactorScore
from harness.ledger.ledger import Ledger, LoopView, Trial
from harness.pit import tools
from harness.splits import get as get_split
from loop import config
from loop.score import score_run

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = config.load(ROOT / "configs" / "cn_refine.yaml")          # csi300, transfer to csi500 and sp500
SPLIT = get_split(CFG.split)
RUN = "refine-cn-2018-2023-s0-deadbeef"
WINDOWS = ("feedback", "pilot", "heldout")
TRANSFER = [f"transfer:{u}:{w}" for u in ("csi500", "sp500") for w in WINDOWS]


def fake_evaluate():
    calls = []
    def ev(expr, window, universe, region, quantile=0.2):
        calls.append((expr, window, universe, region))
        ic = len(expr) / 100
        return FactorScore(True, "", ic, 2 * ic, ic / 10, 3 * ic, 0.5, 2, [ic, -ic], ["2022-07-15", "2022-07-18"])
    ev.calls = calls
    return ev


def make_ledger(tmp_path, exprs=("$close", "Ref($close, -1)", "Mean($close, 5)"), run=RUN):
    """One trial per expression; the second is invalid. Valid ones carry feedback and pilot, as the arms leave them."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ids = []
    for step, e in enumerate(exprs):
        valid = "-1" not in e
        tid = ledger.append(Trial(arm="refine", run=run, step=step, artifact=e, valid=valid,
                                  error="" if valid else "reads the future"))
        ids.append(tid)
        if valid:
            for w in ("feedback", "pilot"): ledger.score(tid, w, {"ok": True, "ic": 0.01, "daily_ls": [], "dates": []})
    return ledger, ids


def windows_of(ledger, tid):
    return sorted(ledger.trials()[tid]["scores"])


def test_score_run_writes_heldout_and_transfer_for_every_valid_trial(tmp_path):
    ledger, ids = make_ledger(tmp_path)
    ev = fake_evaluate()
    n = score_run(CFG, RUN, ledger, evaluate=ev)
    for tid in (ids[0], ids[2]):
        assert windows_of(ledger, tid) == sorted(["feedback", "pilot", "heldout", *TRANSFER])
        rec = ledger.trials()[tid]["scores"]["transfer:sp500:heldout"]
        assert rec["ok"] is True and rec["daily_ls"] == [rec["ic"], -rec["ic"]] and rec["dates"][0] == "2022-07-15"
    assert n == len(ev.calls) == 2 * (1 + len(TRANSFER))


def test_invalid_trials_are_skipped(tmp_path):
    ledger, ids = make_ledger(tmp_path)
    ev = fake_evaluate()
    score_run(CFG, RUN, ledger, evaluate=ev)
    assert ledger.trials()[ids[1]]["scores"] == {}
    assert all(e != "Ref($close, -1)" for e, *_ in ev.calls)


def test_universe_window_and_region_passed_to_evaluate(tmp_path):
    ledger, _ = make_ledger(tmp_path, exprs=("$close",))
    ev = fake_evaluate()
    score_run(CFG, RUN, ledger, evaluate=ev)
    assert ev.calls == [("$close", SPLIT.heldout, "csi300", "cn"),
                        ("$close", SPLIT.feedback, "csi500", "cn"), ("$close", SPLIT.pilot, "csi500", "cn"),
                        ("$close", SPLIT.heldout, "csi500", "cn"),
                        ("$close", SPLIT.feedback, "sp500", "us"), ("$close", SPLIT.pilot, "sp500", "us"),
                        ("$close", SPLIT.heldout, "sp500", "us")]


def test_universes_outer_trials_inner_so_the_region_changes_once(tmp_path):
    ledger, _ = make_ledger(tmp_path, exprs=("$close", "Mean($close, 5)", "$open", "$high"))
    cfg = replace(CFG, transfer_to=("sp500", "csi500"))
    ev = fake_evaluate()
    score_run(cfg, RUN, ledger, evaluate=ev)
    universes = [u for _, _, u, _ in ev.calls]
    blocks = [u for i, u in enumerate(universes) if i == 0 or u != universes[i - 1]]
    assert blocks == ["csi300", "sp500", "csi500"]                      # each universe is one contiguous block
    regions = [r for *_, r in ev.calls]
    assert sum(a != b for a, b in itertools.pairwise(regions)) == 2       # cn -> us -> cn, never per trial


def test_second_call_writes_nothing(tmp_path):
    ledger, _ = make_ledger(tmp_path)
    score_run(CFG, RUN, ledger, evaluate=fake_evaluate())
    before = ledger.path.read_bytes()
    ev = fake_evaluate()
    assert score_run(CFG, RUN, ledger, evaluate=ev) == 0
    assert ev.calls == [] and ledger.path.read_bytes() == before


def test_partly_scored_run_is_completed_without_rewriting(tmp_path):
    ledger, ids = make_ledger(tmp_path)
    ledger.score(ids[0], "heldout", {"ok": True, "ic": 0.99, "daily_ls": [], "dates": []})
    ledger.score(ids[2], "transfer:csi500:pilot", {"ok": False, "ic": None, "daily_ls": [], "dates": []})
    before = ledger.path.read_bytes()
    ev = fake_evaluate()
    n = score_run(CFG, RUN, ledger, evaluate=ev)
    assert n == len(ev.calls) == 2 * (1 + len(TRANSFER)) - 2
    assert ledger.path.read_bytes().startswith(before)                  # append-only: what was there is untouched
    t = ledger.trials()
    assert t[ids[0]]["scores"]["heldout"]["ic"] == 0.99 and t[ids[2]]["scores"]["transfer:csi500:pilot"]["ok"] is False
    assert ("$close", SPLIT.heldout, "csi300", "cn") not in ev.calls
    assert ("Mean($close, 5)", SPLIT.pilot, "csi500", "cn") not in ev.calls
    for tid in (ids[0], ids[2]):
        assert windows_of(ledger, tid) == sorted(["feedback", "pilot", "heldout", *TRANSFER])


def test_trials_of_another_run_are_untouched(tmp_path):
    ledger, _ = make_ledger(tmp_path)
    other = ledger.append(Trial(arm="refine", run="refine-cn-2018-2023-s1-cafef00d", step=0, artifact="$volume"))
    ev = fake_evaluate()
    score_run(CFG, RUN, ledger, evaluate=ev)
    assert ledger.trials()[other]["scores"] == {} and all(e != "$volume" for e, *_ in ev.calls)


def test_loop_view_still_shows_only_feedback(tmp_path):
    ledger, ids = make_ledger(tmp_path)
    score_run(CFG, RUN, ledger, evaluate=fake_evaluate())
    view = LoopView(ledger).trials()
    assert {w for t in view.values() for w in t["scores"]} == {"feedback"}
    assert "transfer:csi500:heldout" in ledger.trials()[ids[0]]["scores"]


@pytest.mark.parametrize("universe", ["csi500", "sp500"])
def test_home_universe_region(tmp_path, universe):
    ledger, _ = make_ledger(tmp_path, exprs=("$close",))
    ev = fake_evaluate()
    score_run(replace(CFG, universe=universe, transfer_to=()), RUN, ledger, evaluate=ev)
    assert ev.calls == [("$close", SPLIT.heldout, universe, config.region_of(universe))]


def test_missing_data_propagates_out_of_score_run(tmp_path, monkeypatch):
    """The real evaluate against an empty data root: the scorer stops, and no score is written for anyone."""
    monkeypatch.setattr(tools, "QLIB_DATA", tmp_path / "empty")
    monkeypatch.setattr(backtest, "_MEMO", {})
    ledger, _ = make_ledger(tmp_path)
    before = ledger.path.read_bytes()
    with pytest.raises(FileNotFoundError, match="cn_data"):
        score_run(CFG, RUN, ledger)
    assert ledger.path.read_bytes() == before
