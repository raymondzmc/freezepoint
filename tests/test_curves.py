"""Tests for the refinement-count curve, the selection statistics and the report. No data: every run directory is
synthetic, written by `make_run` with the real writers (`config.run_dir`, `Ledger.append`, `Ledger.score`,
`score_record`), so a change in the on-disk format breaks these tests."""
import json
import math
import pathlib
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis import report as rep
from analysis.curves import curve, paired_difference, selection_stats
from harness.backtest import FactorScore, score_record
from harness.ledger.ledger import Ledger, Trial
from harness.ledger.stats import deflated_sharpe, pbo_cscv
from loop import config, provenance

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = config.load(ROOT / "configs" / "cn_refine.yaml")       # csi300, transfer to csi500 and sp500
DATES = [str(d.date()) for d in pd.bdate_range("2018-01-02", periods=40)]
PROVENANCE = {}       # one computation per split: the real one, cached, since every synthetic run needs it
UNSCORED = object()     # a trial with no scores at all, as an invalid candidate is left
nan = float("nan")


def cfg_for(arm, max_steps=3, provider="api"):
    g = BASE.generator
    if provider == "fake":
        g = replace(g, provider="fake", model="fake", revision="fake", exprs=("$close",))
    return config.validate(replace(BASE, arm=arm, generator=g,
                                   budget=replace(BASE.budget, max_steps=max_steps, n_candidates=max_steps + 1)))


def trial(fb=UNSCORED, ho=0.0, valid=True, ls=None, dates=None, cost=None, tr=None):
    """One candidate: feedback ic `fb` (None: a failed score), held-out ic `ho` (None: failed), its daily long-short
    series `ls` on `dates` (defaults: a random series on all of DATES), the generation's `cost` dict, and `tr`, the
    held-out ic per transfer universe (default: csi500 gets `ho`, sp500 nothing)."""
    return {"fb": fb, "ho": ho, "valid": valid, "ls": ls, "dates": dates, "cost": cost, "tr": tr}


NAMES = {"csi500": 480.0, "sp500": 360.0}    # names per day each transfer universe's synthetic scores record


def record(ic, ls=None, dates=None, n_names=nan):
    if ic is None:
        return score_record(FactorScore(False, "failed on purpose", nan, nan, nan, nan, nan, 0, [], []))
    ls = list(ls) if ls is not None else [ic] * len(DATES)
    dates = list(dates) if dates is not None else DATES[:len(ls)]
    return score_record(FactorScore(True, "", ic, 2 * ic, ic / 10, 3 * ic, 0.5, len(ls), ls, dates, n_names))


def make_run(root, arm, seed, trials, *, cfg=None, chosen="best", done=True, **cfg_kw):
    """A run directory as `loop.run` leaves it. `chosen="best"` picks the first highest feedback ic among the valid
    trials with an ok feedback score, by hand; `chosen=None` writes a null id."""
    cfg = cfg or cfg_for(arm, **cfg_kw)
    d = config.run_dir(cfg, seed, root)
    if cfg.split not in PROVENANCE: PROVENANCE[cfg.split] = provenance.compute(cfg)
    (d / "provenance.json").write_text(json.dumps(PROVENANCE[cfg.split], indent=2) + "\n")
    rid = config.run_id(cfg, seed)
    ledger = Ledger(d / "ledger.jsonl")
    (d / "calls.jsonl").write_text("")
    rng = np.random.default_rng(1000 * seed + 7)
    best, best_ic = None, None
    for step, t in enumerate(trials):
        expr = f"Mean($close, {step + 2})"
        tid = ledger.append(Trial(arm=arm, run=rid, step=step, artifact=expr, valid=t["valid"],
                                  error="" if t["valid"] else "rejected on purpose", cost=t.get("cost") or {}))
        if t["fb"] is UNSCORED: continue
        ls = t["ls"] if t["ls"] is not None else rng.normal(0.001, 0.01, len(t["dates"] or DATES))
        ledger.score(tid, "feedback", record(t["fb"], ls, t["dates"]))
        ledger.score(tid, "pilot", record(t["fb"]))
        ledger.score(tid, "heldout", record(t["ho"]))
        for u, v in (t.get("tr") or {"csi500": t["ho"]}).items():
            ledger.score(tid, f"transfer:{u}:heldout", record(v, n_names=NAMES[u]))
        if t["valid"] and t["fb"] is not None and (best is None or t["fb"] > best_ic):
            best, best_ic = (tid, expr, step), t["fb"]
    pick = best if chosen == "best" and best else (None, None, None)
    (d / "chosen.json").write_text(json.dumps({"id": pick[0], "artifact": pick[1], "arm": arm, "run": rid,
                                               "seed": seed, "step": pick[2]}, indent=2) + "\n")
    if done: (d / "done").write_text("2026-11-10T00:00:00+00:00\n")
    return d


def rows(df, arm, col="mean"):
    return df[df.arm == arm].sort_values("step")[col].to_numpy()


def same(a, b):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0, atol=1e-12)


# ---- the brief's four tests -----------------------------------------------------------------------------------

def test_curve_bestsofar_is_monotone_on_feedback_and_not_on_heldout(tmp_path):
    dirs = []
    for arm in ("refine", "bestofn"):
        for seed, off in enumerate((0.0, 0.002, -0.002)):
            ts = [trial(fb + off, ho + off) for fb, ho in zip((0.01, 0.02, 0.015, 0.03), (0.02, 0.04, 0.0, 0.01))]
            dirs.append(make_run(tmp_path, arm, seed, ts))
    fb, ho = curve(dirs, window="feedback"), curve(dirs, window="heldout")
    for arm in ("refine", "bestofn"):
        same(rows(fb, arm), [0.01, 0.02, 0.02, 0.03])
        assert np.all(np.diff(rows(fb, arm)) >= 0)
        same(rows(ho, arm), [0.02, 0.04, 0.04, 0.01])
        assert np.any(np.diff(rows(ho, arm)) < 0)
    assert list(fb.columns) == ["arm", "step", "mean", "lo", "hi", "n"]
    assert sorted(set(fb.step)) == [0, 1, 2, 3] and set(fb.n) == {3}


def test_frozen_is_flat(tmp_path):
    dirs = [make_run(tmp_path, "frozen", s, [trial(0.02, ho)]) for s, ho in enumerate((0.01, 0.02, 0.04))]
    c = curve(dirs)
    assert list(rows(c, "frozen", "step")) == [0, 1, 2, 3]
    for col in ("mean", "lo", "hi", "n"):
        assert len(set(rows(c, "frozen", col))) == 1
    same(rows(c, "frozen"), [0.07 / 3] * 4)


def test_ci_width_shrinks_with_more_seeds(tmp_path):
    dirs = [make_run(tmp_path, "refine", s, [trial(0.01, ho)] * 4)
            for s, ho in enumerate((0.01, 0.03, 0.01, 0.03, 0.02))]
    width = {k: (rows(c, "refine", "hi") - rows(c, "refine", "lo"))
             for k, c in ((2, curve(dirs[:2])), (5, curve(dirs)))}
    assert np.all(width[5] < width[2])


def test_report_writes_four_files(tmp_path):
    dirs = [make_run(tmp_path / "runs", arm, s, [trial(0.01 * (k + 1), 0.01) for k in range(1 if arm == "frozen" else 4)])
            for arm in ("frozen", "refine", "bestofn") for s in (0, 1)]
    out = tmp_path / "out"
    rep.report(dirs, out)
    assert {p.name for p in out.iterdir()} == {"curve.csv", "selection.csv", "report.md", "cost.md"}
    c = pd.read_csv(out / "curve.csv")
    assert list(c.columns) == ["window", "arm", "step", "mean", "lo", "hi", "n"]
    assert list(dict.fromkeys(c.window)) == ["feedback", "pilot", "heldout", "transfer:csi500:heldout",
                                             "transfer:sp500:heldout"]
    assert len(c) == 5 * 3 * 4
    s = pd.read_csv(out / "selection.csv")
    assert list(s.columns) == ["run", "arm", "seed", "n_trials", "n_valid", "dsr_of_chosen", "pbo"] and len(s) == 6


# ---- the selection rule ---------------------------------------------------------------------------------------

def test_selection_is_by_feedback_ic_with_ties_to_the_earliest(tmp_path):
    ts = [trial(0.9, 0.9, valid=False),         # invalid (scored here on purpose): never the best
          trial(None, 0.8),                     # failed feedback score: never the best
          trial(0.01, 0.10),
          trial(0.01, 0.20),                    # ties with step 2: the earlier one stays
          trial(0.005, 0.99),                   # better held-out, worse feedback: not chosen
          trial(0.02, None)]                    # chosen, but its held-out score failed
    other = [trial(0.01 * (k + 1), 0.05) for k in range(6)]
    dirs = [make_run(tmp_path, "refine", 0, ts, max_steps=5), make_run(tmp_path, "refine", 1, other, max_steps=5)]
    ho = curve(dirs[:1])
    same(rows(ho, "refine"), [nan, nan, 0.10, 0.10, 0.10, nan])
    assert list(rows(ho, "refine", "n")) == [0, 0, 1, 1, 1, 0]
    same(rows(curve(dirs[:1], window="feedback"), "refine"), [nan, nan, 0.01, 0.01, 0.01, 0.02])
    same(rows(curve(dirs[:1], metric="icir"), "refine"), [nan, nan, 0.20, 0.20, 0.20, nan])
    both = curve(dirs)
    assert list(rows(both, "refine", "n")) == [1, 1, 2, 2, 2, 1]
    same(rows(both, "refine")[[0, 2, 5]], [0.05, 0.075, 0.05])


def test_interval_matches_a_hand_computation(tmp_path):
    vals = (0.01, 0.04, 0.02)
    dirs = [make_run(tmp_path, "refine", s, [trial(0.01, v)] * 4) for s, v in enumerate(vals)]
    c = curve(dirs)
    m, sd = np.mean(vals), np.std(vals, ddof=1)
    h = stats.t.ppf(0.975, 2) * sd / math.sqrt(3)
    same(rows(c, "refine"), [m] * 4)
    same(rows(c, "refine", "lo"), [m - h] * 4)
    same(rows(c, "refine", "hi"), [m + h] * 4)
    one = curve(dirs[:1])
    same(rows(one, "refine"), [0.01] * 4)
    assert np.isnan(rows(one, "refine", "lo")).all() and np.isnan(rows(one, "refine", "hi")).all()
    assert list(rows(one, "refine", "n")) == [1] * 4


# ---- inputs must be one finished cell ---------------------------------------------------------------------------

def test_unfinished_run_raises(tmp_path):
    ok = make_run(tmp_path, "refine", 0, [trial(0.01)] * 4)
    bad = make_run(tmp_path, "refine", 1, [trial(0.01)] * 4, done=False)
    for f in (curve, selection_stats, lambda d: rep.report(d, tmp_path / "out")):
        with pytest.raises(ValueError, match=bad.name):
            f([ok, bad])


def test_two_cells_mixed_raise_and_name_the_fields(tmp_path):
    a = make_run(tmp_path, "refine", 0, [trial(0.01)] * 4)
    b = make_run(tmp_path, "bestofn", 0, [trial(0.01)] * 5, max_steps=4)
    c = make_run(tmp_path, "frozen", 0, [trial(0.01)], provider="fake")
    with pytest.raises(ValueError, match=r"budget\.max_steps") as e:
        curve([a, b])
    assert a.name in str(e.value) and b.name in str(e.value)
    with pytest.raises(ValueError, match=r"generator\.provider"):
        selection_stats([a, c])


def test_duplicated_arm_and_seed_raises(tmp_path):
    a = make_run(tmp_path, "refine", 0, [trial(0.01)] * 4)
    copy = tmp_path / "elsewhere" / a.name
    shutil.copytree(a, copy)
    with pytest.raises(ValueError, match="seed 0"):
        curve([a, copy])
    with pytest.raises(ValueError, match="seed 0"):
        curve([a, a])


def extra_trial(d, step, artifact="Std($close, 9)"):
    """Append one more trial of `d`'s run to its ledger, as a second live loop would have."""
    Ledger(d / "ledger.jsonl").append(Trial(arm=d.name.split("-")[0], run=d.name, step=step, artifact=artifact))
    return d


@pytest.mark.parametrize("name, build, found", [
    ("too few", lambda root: make_run(root, "refine", 0, [trial(0.01)] * 3), "3 trials"),
    ("too many", lambda root: make_run(root, "refine", 0, [trial(0.01)] * 5), "5 trials"),
    ("duplicated step, too many", lambda root: extra_trial(make_run(root, "refine", 0, [trial(0.01)] * 4), 2),
     "steps [0, 1, 2, 2, 3]"),
    ("duplicated step, right count", lambda root: extra_trial(make_run(root, "refine", 0, [trial(0.01)] * 3), 1),
     "steps [0, 1, 1, 2]"),
    ("frozen with two", lambda root: make_run(root, "frozen", 0, [trial(0.01)] * 2), "2 trials"),
    ("best-of-N with three", lambda root: make_run(root, "bestofn", 0, [trial(0.01)] * 3), "3 trials"),
])
def test_a_run_whose_trials_do_not_match_its_budget_is_refused(tmp_path, name, build, found):
    d = build(tmp_path)
    for f in (curve, selection_stats):
        with pytest.raises(ValueError, match=d.name) as e:
            f([d])
        assert found in str(e.value), str(e.value)


def test_a_run_that_matches_its_budget_loads(tmp_path):
    dirs = [make_run(tmp_path, "frozen", 0, [trial(0.01)]), make_run(tmp_path, "refine", 0, [trial(0.01)] * 4),
            make_run(tmp_path, "bestofn", 0, [trial(0.01)] * 4)]
    assert set(selection_stats(dirs).n_trials) == {1, 4}


def test_a_run_without_provenance_is_refused(tmp_path):
    a = make_run(tmp_path, "frozen", 0, [trial(0.01)])
    b = make_run(tmp_path, "frozen", 1, [trial(0.01)])
    (b / "provenance.json").unlink()
    for f in (curve, selection_stats):
        with pytest.raises(ValueError, match=f"{b.name}.*provenance.json"):
            f([a, b])


def edit_provenance(d, **changes):
    p = json.loads((d / "provenance.json").read_text())
    p.update(changes)
    (d / "provenance.json").write_text(json.dumps(p))


@pytest.mark.parametrize("field, value", [
    ("prompts_sha256", "0" * 64), ("source_sha256", "1" * 64),
    ("lock_files", {"uv.lock": "2" * 64}),
    ("split", {"name": "cn-2018-2023", "feedback": ["2018-01-01", "2021-06-30"], "pilot": ["2022-01-17", "2022-06-30"],
               "heldout": ["2022-07-15", "2023-12-31"], "forward": None, "embargo_days": 10}),
])
def test_runs_that_disagree_on_provenance_are_two_cells(tmp_path, field, value):
    a = make_run(tmp_path, "frozen", 0, [trial(0.01)])
    b = make_run(tmp_path, "refine", 0, [trial(0.01)] * 4)
    edit_provenance(b, **{field: value})
    with pytest.raises(ValueError, match=field) as e:
        curve([a, b])
    assert a.name in str(e.value) and b.name in str(e.value)


def test_git_commit_and_dirty_flag_are_recorded_not_compared(tmp_path):
    a = make_run(tmp_path, "frozen", 0, [trial(0.01)])
    b = make_run(tmp_path, "refine", 0, [trial(0.01)] * 4)
    edit_provenance(b, git_commit="f" * 40, git_dirty=True)
    assert len(curve([a, b]))


def test_no_run_dirs_raises():
    with pytest.raises(ValueError):
        curve([])


# ---- selection statistics ------------------------------------------------------------------------------------

def test_n_trials_counts_invalid_candidates(tmp_path):
    d = make_run(tmp_path, "refine", 0, [trial(0.01), trial(valid=False), trial(None), trial(0.02)])
    s = selection_stats([d]).iloc[0]
    assert (s.run, s.arm, s.seed, s.n_trials, s.n_valid) == (d.name, "refine", 0, 4, 2)


def test_pbo_is_nan_for_a_frozen_run(tmp_path):
    s = selection_stats([make_run(tmp_path, "frozen", 0, [trial(0.01)])]).iloc[0]
    assert np.isnan(s.pbo) and np.isfinite(s.dsr_of_chosen)


def test_pbo_aligns_trials_on_common_dates(tmp_path):
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 0.01, 30), rng.normal(0, 0.01, 35)
    d = make_run(tmp_path, "refine", 0, [trial(0.01, ls=a, dates=DATES[:30]), trial(0.02, ls=b, dates=DATES[5:])],
                 max_steps=1)
    expected = pbo_cscv(np.column_stack([a[5:30], b[:25]]), S=8)["pbo"]
    assert selection_stats([d]).iloc[0].pbo == expected
    short = make_run(tmp_path, "refine", 1, [trial(0.01, ls=a[:10], dates=DATES[:10]),
                                             trial(0.02, ls=b[:10], dates=DATES[5:15])], max_steps=1)
    assert np.isnan(selection_stats([short]).iloc[0].pbo)        # 5 common days, fewer than the 8 blocks


def test_dsr_of_chosen_matches_a_direct_call(tmp_path):
    rng = np.random.default_rng(11)
    series = [rng.normal(mu, 0.01, 40) for mu in (0.0005, 0.002, 0.001)]
    ts = [trial(0.01, ls=series[0]), trial(valid=False), trial(0.03, ls=series[1]), trial(0.02, ls=series[2])]
    d = make_run(tmp_path, "refine", 0, ts)
    x = series[1]
    srs = [s.mean() / s.std(ddof=1) for s in series]
    c = x - x.mean()
    m2, m3, m4 = (c ** 2).mean(), (c ** 3).mean(), (c ** 4).mean()
    expected = deflated_sharpe(srs[1], 4, np.var(srs, ddof=1), 40, m3 / m2 ** 1.5, m4 / m2 ** 2)
    assert selection_stats([d]).iloc[0].dsr_of_chosen == pytest.approx(expected, rel=1e-12)
    none = make_run(tmp_path, "refine", 1, ts, chosen=None)
    assert np.isnan(selection_stats([none]).iloc[0].dsr_of_chosen)


# ---- the report ------------------------------------------------------------------------------------------------

# Held-out rank IC per seed at steps 0..3 (feedback rises 0.01..0.04, so every step's candidate is the one held).
FALL = {0: (0.020, 0.050, 0.030, 0.000), 1: (0.021, 0.051, 0.031, 0.003), 2: (0.019, 0.049, 0.029, -0.002)}
FLAT = {0: (0.03, 0.04, 0.02, 0.035), 1: (0.03, 0.02, 0.04, 0.025), 2: (0.04, 0.03, 0.03, 0.03),
        3: (0.02, 0.03, 0.03, 0.03)}


def refine_cell(root, held, provider="api", arms=("frozen", "refine", "bestofn")):
    """One run per arm and seed of `held`; feedback rises 0.01..0.04 and held-out follows `held[seed]`."""
    dirs = []
    for arm in arms:
        for s, ho in held.items():
            ts = [trial(fb, h) for fb, h in zip((0.01, 0.02, 0.03, 0.04), ho)]
            dirs.append(make_run(root, arm, s, ts[:1] if arm == "frozen" else ts, provider=provider))
    return dirs


def statement(md):
    return md.split("## Refine arm")[1].split("\n## ")[0]


def para(md, label):
    """The statement's paragraph that starts with `label`."""
    (p,) = [p for p in statement(md).split("\n\n") if p.startswith(label)]
    return p


def report_md(tmp_path, dirs):
    rep.report(dirs, tmp_path / "out")
    return (tmp_path / "out" / "report.md").read_text()


def test_paired_difference_matches_a_hand_computation():
    per_run = [[0.1, 0.2, 0.05], [0.3, nan, 0.1], [0.2, 0.25, 0.3], [nan, 0.1, 0.1]]
    p = paired_difference(per_run, 0, 2)
    d = np.array([0.05, 0.2, -0.1])
    h = stats.t.ppf(0.975, 2) * d.std(ddof=1) / math.sqrt(3)
    same([p.mean, p.lo, p.hi, p.half_width], [d.mean(), d.mean() - h, d.mean() + h, h])
    assert (p.n, p.left_out) == (3, 1)
    q = paired_difference(per_run, 1, 2)
    assert (q.n, q.left_out) == (3, 1) and q.mean == pytest.approx(np.mean([0.15, -0.05, 0.0]))
    one = paired_difference(per_run[:2], 1, 2)
    assert (one.n, one.left_out) == (1, 1) and np.isnan(one.lo) and np.isnan(one.hi)


def test_report_says_held_out_fell_past_the_half_width(tmp_path):
    md = report_md(tmp_path, refine_cell(tmp_path / "runs", FALL))
    peak, start = para(md, "Peak against final"), para(md, "Start against final")
    d = np.array([FALL[s][1] - FALL[s][3] for s in FALL])
    h = stats.t.ppf(0.975, 2) * d.std(ddof=1) / math.sqrt(3)
    assert f"averages {d.mean():.4f} over 3 seeds" in peak and f"half-width {h:.4f}" in peak
    assert "is below its peak (step 1," in peak and "left out" not in peak
    assert "is below its step-0 value by more than the half-width" in start
    fb = para(md, "Feedback")
    assert "0.0100 at step 0 and 0.0400 at step 3" in fb and "averages 0.0300 over 3 seeds" in fb
    assert "overstates the drop" in statement(md)


def test_report_says_held_out_did_not_fall_past_the_half_width(tmp_path):
    md = report_md(tmp_path, refine_cell(tmp_path / "runs", FLAT))
    assert "is not below its peak" in para(md, "Peak against final")
    assert "is not below its step-0 value" in para(md, "Start against final")


def test_a_seed_leaving_the_average_is_not_a_drop(tmp_path):
    """The reviewer's scenario A: held-out never changes for any seed, but seed 0's last candidate has a failed
    held-out score, so the mean at step 3 loses seed 0's high value."""
    const = {0: 0.10, 1: 0.02, 2: 0.01, 3: 0.015, 4: 0.02}
    dirs = [make_run(tmp_path / "runs", "refine", s,
                     [trial(fb, None if (s == 0 and k == 3) else v) for k, fb in enumerate((0.01, 0.02, 0.03, 0.04))])
            for s, v in const.items()]
    md = report_md(tmp_path, dirs)
    for label, what in (("Peak against final", "its peak"), ("Start against final", "its step-0 value")):
        p = para(md, label)
        assert "averages 0.0000 over 4 seeds" in p and f"is not below {what}" in p
        assert "1 of the refine arm's 5 seeds was left out because its held-out value at step 0 or at step 3" in p
    assert "is below its" not in statement(md)


def test_a_seed_without_a_step_0_value_is_left_out_of_start_against_final(tmp_path):
    ts = {s: [trial(fb, h) for fb, h in zip((0.01, 0.02, 0.03, 0.04), FALL[s])] for s in FALL}
    ts[2][0] = trial(valid=False)
    md = report_md(tmp_path, [make_run(tmp_path / "runs", "refine", s, t) for s, t in ts.items()])
    start = para(md, "Start against final")
    assert "over 2 seeds" in start
    assert "1 of the refine arm's 3 seeds was left out because its held-out value at step 0 or at step 3" in start
    assert "left out" not in para(md, "Peak against final")
    assert "1 of the refine arm's 3 seeds was left out because its feedback value" in para(md, "Feedback")


def test_report_cannot_answer_with_fewer_than_two_refine_seeds(tmp_path):
    s = statement(report_md(tmp_path / "one", refine_cell(tmp_path / "a", {0: FALL[0]})))
    assert "cannot be answered" in s and "the refine arm has only one seed" in s and "is below its" not in s
    s = statement(report_md(tmp_path / "none", refine_cell(tmp_path / "b", FALL, arms=("frozen",))))
    assert "cannot be answered" in s and "no refine arm" in s


def test_report_cannot_answer_when_held_out_at_the_last_step_has_one_seed(tmp_path):
    ts = [trial(0.01 * (k + 1), 0.02) for k in range(4)]
    dirs = [make_run(tmp_path / "runs", "refine", 0, ts), make_run(tmp_path / "runs", "refine", 1, [*ts[:3], trial(0.05, None)])]
    md = report_md(tmp_path, dirs)
    for label in ("Peak against final", "Start against final"):
        p = para(md, label)
        assert "cannot be answered" in p and "only one seed of the refine arm's 2 has a held-out value at both step 0 " \
            "and step 3" in p


def transfer_cell(root, home, csi500, sp500=None):
    """Refine and frozen runs per seed: feedback rises 0.01..0.04; held-out follows `home[seed]` and the transfer
    windows `csi500[seed]` and `sp500[seed]` (absent: no sp500 score)."""
    dirs = []
    for arm in ("frozen", "refine"):
        for s in home:
            ts = [trial(fb, home[s][k], tr={"csi500": csi500[s][k], **({"sp500": sp500[s][k]} if sp500 else {})})
                  for k, fb in enumerate((0.01, 0.02, 0.03, 0.04))]
            dirs.append(make_run(root, arm, s, ts[:1] if arm == "frozen" else ts))
    return dirs


def section(md, heading):
    return md.split(f"## {heading}\n")[1].split("\n## ")[0]


def test_report_gives_the_paired_statements_for_each_transfer_universe(tmp_path):
    md = report_md(tmp_path, transfer_cell(tmp_path / "runs", FALL, FALL, FLAT))
    heads = [ln[3:] for ln in md.splitlines() if ln.startswith("## ")]
    i = heads.index("Refine arm: does held-out fall while feedback rises?")
    assert heads[i + 1:i + 3] == ["Transfer to csi500: held-out window", "Transfer to sp500: held-out window"]
    assert heads[i + 3] == "Held-out curve"
    home = statement(md)
    assert "csi500" not in home and "sp500" not in home and "Start against final" in home
    a, b = section(md, heads[i + 1]), section(md, heads[i + 2])
    assert "480.0 names per day" in a and "360.0 names per day" in b
    peak = next(p for p in a.split("\n\n") if p.startswith("Peak against final"))
    d = np.array([FALL[s][1] - FALL[s][3] for s in FALL])
    h = stats.t.ppf(0.975, 2) * d.std(ddof=1) / math.sqrt(3)
    assert f"the csi500 held-out rank IC at step 1 minus that at step 3 averages {d.mean():.4f} over 3 seeds" in peak
    assert f"half-width {h:.4f}" in peak and "The csi500 held-out score at step 3 is below its peak (step 1," in peak
    start = next(p for p in a.split("\n\n") if p.startswith("Start against final"))
    assert "is below its step-0 value by more than the half-width" in start
    assert "is not below its step-0 value" in next(p for p in b.split("\n\n") if p.startswith("Start against final"))
    assert "Feedback:" not in a                         # the feedback paragraph is the home universe's alone


def test_transfer_statement_without_values_cannot_be_answered(tmp_path):
    md = report_md(tmp_path, transfer_cell(tmp_path / "runs", FALL, FALL))
    b = section(md, "Transfer to sp500: held-out window")
    assert "cannot be answered" in b and "no refine seed has a sp500 held-out value at any step" in b
    assert "No candidate has a sp500 held-out score that records its names per day." in b


def test_report_flags_the_fake_generator(tmp_path):
    rep.report(refine_cell(tmp_path / "fake", FALL, provider="fake"), tmp_path / "f")
    lines = [ln for ln in (tmp_path / "f" / "report.md").read_text().splitlines() if ln.strip()]
    assert lines[0].startswith("# ") and "fake generator" in lines[1] and "not a result" in lines[1]
    rep.report(refine_cell(tmp_path / "api", FALL), tmp_path / "a")
    assert "fake generator" not in (tmp_path / "a" / "report.md").read_text()


def test_command_line(tmp_path, capsys):
    dirs = refine_cell(tmp_path / "runs", FALL)
    out = tmp_path / "out"
    assert rep.main([*map(str, dirs), "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(out / "report.md")
    assert (out / "report.md").is_file()
    (dirs[0] / "done").unlink()
    assert rep.main([*map(str, dirs), "--out", str(tmp_path / "bad")]) != 0
    assert dirs[0].name in capsys.readouterr().err


# ---- the cost summary -----------------------------------------------------------------------------------------

def cost_rows(path):
    """`cost.md`'s table as {first cell: [other cells]}."""
    lines = [ln for ln in path.read_text().splitlines() if ln.startswith("|")][2:]
    return {cells[0]: cells[1:] for cells in ([c.strip() for c in ln.strip("|").split("|")] for ln in lines)}


def test_cost_md_sums_the_cost_of_every_trial_per_arm(tmp_path):
    c = lambda i, o, s: {"input_tokens": i, "output_tokens": o, "seconds": s}
    dirs = [make_run(tmp_path / "runs", "frozen", s, [trial(0.01, 0.01, cost=c(100, 10, 1.5))]) for s in (0, 1)]
    dirs += [make_run(tmp_path / "runs", "refine", s, [trial(0.01 * (k + 1), 0.01, cost=c(200, 20, 2.0))
                                                      for k in range(4)]) for s in (0, 1, 2)]
    dirs += [make_run(tmp_path / "runs", "bestofn", 0, [trial(0.01, 0.01, cost=c(50, 5, 0.25)) for _ in range(4)])]
    rep.report(dirs, tmp_path / "out")
    path = tmp_path / "out" / "cost.md"
    t = cost_rows(path)
    assert list(t) == ["frozen", "refine", "best-of-N", "total"]
    assert t["frozen"] == ["2", "2", "200", "20", "3.0"]
    assert t["refine"] == ["3", "12", "2400", "240", "24.0"]
    assert t["best-of-N"] == ["1", "4", "200", "20", "1.0"]
    assert t["total"] == ["6", "18", "2800", "280", "28.0"]
    text = path.read_text()
    assert "Scoring time is not in them" in text and "generations" in text


def test_cost_md_for_a_fake_cell_is_zeros(tmp_path):
    dirs = [make_run(tmp_path / "runs", arm, s, [trial(0.01 * (k + 1), 0.01) for k in range(1 if arm == "frozen" else 4)],
                     provider="fake") for arm in ("frozen", "refine") for s in (0, 1)]
    rep.report(dirs, tmp_path / "out")
    t = cost_rows(tmp_path / "out" / "cost.md")
    assert list(t) == ["frozen", "refine", "total"]      # arms that are absent get no row
    assert t["frozen"] == ["2", "2", "0", "0", "0.0"] and t["refine"] == ["2", "8", "0", "0", "0.0"]
    assert t["total"] == ["4", "10", "0", "0", "0.0"]
