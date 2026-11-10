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
from analysis.curves import curve, selection_stats
from harness.backtest import FactorScore, score_record
from harness.ledger.ledger import Ledger, Trial
from harness.ledger.stats import deflated_sharpe, pbo_cscv
from loop import config

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = config.load(ROOT / "configs" / "cn_refine.yaml")       # csi300, transfer to csi500 and sp500
DATES = [str(d.date()) for d in pd.bdate_range("2018-01-02", periods=40)]
UNSCORED = object()     # a trial with no scores at all, as an invalid candidate is left
nan = float("nan")


def cfg_for(arm, max_steps=3, provider="api"):
    g = BASE.generator
    if provider == "fake":
        g = replace(g, provider="fake", model="fake", revision="fake", exprs=("$close",))
    return config.validate(replace(BASE, arm=arm, generator=g,
                                   budget=replace(BASE.budget, max_steps=max_steps, n_candidates=max_steps + 1)))


def trial(fb=UNSCORED, ho=0.0, valid=True, ls=None, dates=None):
    """One candidate: feedback ic `fb` (None: a failed score), held-out ic `ho` (None: failed), its daily long-short
    series `ls` on `dates` (defaults: a random series on all of DATES)."""
    return {"fb": fb, "ho": ho, "valid": valid, "ls": ls, "dates": dates}


def record(ic, ls=None, dates=None):
    if ic is None:
        return score_record(FactorScore(False, "failed on purpose", nan, nan, nan, nan, nan, 0, [], []))
    ls = list(ls) if ls is not None else [ic] * len(DATES)
    dates = list(dates) if dates is not None else DATES[:len(ls)]
    return score_record(FactorScore(True, "", ic, 2 * ic, ic / 10, 3 * ic, 0.5, len(ls), ls, dates))


def make_run(root, arm, seed, trials, *, cfg=None, chosen="best", done=True, **cfg_kw):
    """A run directory as `loop.run` leaves it. `chosen="best"` picks the first highest feedback ic among the valid
    trials with an ok feedback score, by hand; `chosen=None` writes a null id."""
    cfg = cfg or cfg_for(arm, **cfg_kw)
    d = config.run_dir(cfg, seed, root)
    rid = config.run_id(cfg, seed)
    ledger = Ledger(d / "ledger.jsonl")
    (d / "calls.jsonl").write_text("")
    rng = np.random.default_rng(1000 * seed + 7)
    best, best_ic = None, None
    for step, t in enumerate(trials):
        expr = f"Mean($close, {step + 2})"
        tid = ledger.append(Trial(arm=arm, run=rid, step=step, artifact=expr, valid=t["valid"],
                                  error="" if t["valid"] else "rejected on purpose"))
        if t["fb"] is UNSCORED: continue
        ls = t["ls"] if t["ls"] is not None else rng.normal(0.001, 0.01, len(t["dates"] or DATES))
        ledger.score(tid, "feedback", record(t["fb"], ls, t["dates"]))
        ledger.score(tid, "pilot", record(t["fb"]))
        ledger.score(tid, "heldout", record(t["ho"]))
        ledger.score(tid, "transfer:csi500:heldout", record(t["ho"]))
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


def test_report_writes_three_files(tmp_path):
    dirs = [make_run(tmp_path / "runs", arm, s, [trial(0.01 * (k + 1), 0.01) for k in range(1 if arm == "frozen" else 4)])
            for arm in ("frozen", "refine", "bestofn") for s in (0, 1)]
    out = tmp_path / "out"
    rep.report(dirs, out)
    assert {p.name for p in out.iterdir()} == {"curve.csv", "selection.csv", "report.md"}
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
    ok = make_run(tmp_path, "refine", 0, [trial(0.01)])
    bad = make_run(tmp_path, "refine", 1, [trial(0.01)], done=False)
    for f in (curve, selection_stats, lambda d: rep.report(d, tmp_path / "out")):
        with pytest.raises(ValueError, match=bad.name):
            f([ok, bad])


def test_two_cells_mixed_raise_and_name_the_fields(tmp_path):
    a = make_run(tmp_path, "refine", 0, [trial(0.01)])
    b = make_run(tmp_path, "bestofn", 0, [trial(0.01)], max_steps=4)
    c = make_run(tmp_path, "frozen", 0, [trial(0.01)], provider="fake")
    with pytest.raises(ValueError, match=r"budget\.max_steps") as e:
        curve([a, b])
    assert a.name in str(e.value) and b.name in str(e.value)
    with pytest.raises(ValueError, match=r"generator\.provider"):
        selection_stats([a, c])


def test_duplicated_arm_and_seed_raises(tmp_path):
    a = make_run(tmp_path, "refine", 0, [trial(0.01)])
    copy = tmp_path / "elsewhere" / a.name
    shutil.copytree(a, copy)
    with pytest.raises(ValueError, match="seed 0"):
        curve([a, copy])
    with pytest.raises(ValueError, match="seed 0"):
        curve([a, a])


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
    d = make_run(tmp_path, "refine", 0, [trial(0.01, ls=a, dates=DATES[:30]), trial(0.02, ls=b, dates=DATES[5:])])
    expected = pbo_cscv(np.column_stack([a[5:30], b[:25]]), S=8)["pbo"]
    assert selection_stats([d]).iloc[0].pbo == expected
    short = make_run(tmp_path, "refine", 1, [trial(0.01, ls=a[:10], dates=DATES[:10]),
                                             trial(0.02, ls=b[:10], dates=DATES[5:15])])
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

def refine_cell(root, last, jitter, provider="api", seeds=(0, 1, 2), arms=("frozen", "refine", "bestofn")):
    """Feedback rises 0.01..0.04; held-out goes 0.02, 0.05, 0.03, `last`, each plus the seed's jitter."""
    dirs = []
    for arm in arms:
        for s in seeds:
            j = jitter[s]
            ts = [trial(fb, ho + j) for fb, ho in zip((0.01, 0.02, 0.03, 0.04), (0.02, 0.05, 0.03, last))]
            dirs.append(make_run(root, arm, s, ts[:1] if arm == "frozen" else ts, provider=provider))
    return dirs


def statement(md):
    return md.split("## Refine arm")[1].split("\n## ")[0]


def test_report_says_held_out_fell_past_the_half_width(tmp_path):
    rep.report(refine_cell(tmp_path / "runs", 0.0, (0.0, 0.001, -0.001)), tmp_path / "out")
    s = statement((tmp_path / "out" / "report.md").read_text())
    h = stats.t.ppf(0.975, 2) * 0.001 / math.sqrt(3)
    assert "0.0500 at step 1" in s and f"{h:.4f}" in s and "0.0500" in s
    assert "is below its peak by more than" in s and "is not below" not in s
    assert "0.0100 at step 0" in s and "0.0400 at step 3" in s


def test_report_says_held_out_did_not_fall_past_the_half_width(tmp_path):
    rep.report(refine_cell(tmp_path / "runs", 0.045, (0.0, 0.02, -0.02)), tmp_path / "out")
    s = statement((tmp_path / "out" / "report.md").read_text())
    assert "is not below its peak by more than" in s


def test_report_cannot_answer_with_fewer_than_two_refine_seeds(tmp_path):
    rep.report(refine_cell(tmp_path / "a", 0.0, (0.0, 0.001), seeds=(0,)), tmp_path / "one")
    s = statement((tmp_path / "one" / "report.md").read_text())
    assert "cannot be answered" in s and "the refine arm has only one seed" in s and "is below its peak" not in s
    rep.report(refine_cell(tmp_path / "b", 0.0, (0.0, 0.001), seeds=(0, 1), arms=("frozen",)), tmp_path / "none")
    s = statement((tmp_path / "none" / "report.md").read_text())
    assert "cannot be answered" in s and "no refine arm" in s


def test_report_cannot_answer_when_held_out_at_the_last_step_has_one_seed(tmp_path):
    ts = [trial(0.01 * (k + 1), 0.02) for k in range(4)]
    dirs = [make_run(tmp_path / "runs", "refine", 0, ts), make_run(tmp_path / "runs", "refine", 1, [*ts[:3], trial(0.05, None)])]
    rep.report(dirs, tmp_path / "out")
    s = statement((tmp_path / "out" / "report.md").read_text())
    assert "cannot be answered" in s and "only one seed of the refine arm's 2 has a held-out value at step 3" in s


def test_report_flags_the_fake_generator(tmp_path):
    rep.report(refine_cell(tmp_path / "fake", 0.0, (0, 0, 0), provider="fake"), tmp_path / "f")
    lines = [ln for ln in (tmp_path / "f" / "report.md").read_text().splitlines() if ln.strip()]
    assert lines[0].startswith("# ") and "fake generator" in lines[1] and "not a result" in lines[1]
    rep.report(refine_cell(tmp_path / "api", 0.0, (0, 0, 0)), tmp_path / "a")
    assert "fake generator" not in (tmp_path / "a" / "report.md").read_text()


def test_command_line(tmp_path, capsys):
    dirs = refine_cell(tmp_path / "runs", 0.0, (0.0, 0.001, -0.001))
    out = tmp_path / "out"
    assert rep.main([*map(str, dirs), "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(out / "report.md")
    assert (out / "report.md").is_file()
    (dirs[0] / "done").unlink()
    assert rep.main([*map(str, dirs), "--out", str(tmp_path / "bad")]) != 0
    assert dirs[0].name in capsys.readouterr().err
