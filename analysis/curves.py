"""The refinement-count curve and the selection statistics of one cell: the finished runs, one per arm and seed, of
one effective config that differs only in `arm`.

Guarantees. (1) Every function takes only one finished cell: a directory without `done`, runs whose configs differ in
anything but `arm` (and the `seeds` list, which is not part of a run's identity), two runs with the same arm and seed, and a run whose trials are not exactly its arm's budget with
steps `0..n-1` once each (what two live loops appending to one run would leave) each raise ValueError naming the
directories (and the fields or the steps found), so smoke runs can never be averaged with real ones. So do a run
without `provenance.json` and runs whose provenance differs in a field of `loop.provenance.COMPARED` (prompts, source,
lock files, split windows; the git commit is not compared). (2) The candidate a run holds after step `k`
is chosen by the rule the arms use (`loop.arms._best`: the first highest feedback ic among the valid trials with an
ok feedback score), always on feedback ic, whatever window and metric are read off it. (3) A missing, failed or null
score is NaN and is left out of the mean and of `n`; it is never read as zero.

The step axis is the number of refinements spent, `0..max_steps`, the same for every arm, so equal step means equal
generations: refine trial `k` is refinement `k`, best-of-N trial `k` is proposal `k`, and frozen has only step 0."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from harness.ledger.ledger import Ledger
from harness.ledger.stats import deflated_sharpe, pbo_cscv
from loop import config, provenance
from loop.arms import _best, _eligible

PBO_BLOCKS = 8
CURVE_COLUMNS = ["arm", "step", "mean", "lo", "hi", "n"]
SELECTION_COLUMNS = ["run", "arm", "seed", "n_trials", "n_valid", "dsr_of_chosen", "pbo"]


@dataclass(frozen=True)
class Run:
    path: Path
    cfg: config.RunConfig
    run: str
    seed: int
    chosen: str | None      # trial id from chosen.json; None when nothing was chosen
    trials: list[dict]      # this run's trials in generation order, full view
    n_trials: int           # every candidate, invalid ones included
    provenance: dict        # provenance.json

    @property
    def arm(self) -> str: return self.cfg.arm


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(_flat(v, f"{prefix}{k}."))
        else: out[f"{prefix}{k}"] = v
    return out


def _load_run(d: Path) -> Run:
    try:
        cfg = config.load(d / "config.yaml")
        chosen = json.loads((d / "chosen.json").read_text())
    except OSError as e:
        raise ValueError(f"{d}: not a complete run directory: {e}") from e
    if not (d / provenance.FILE).is_file():
        raise ValueError(f"{d}: has no {provenance.FILE}, so what code, prompts and data produced it is unknown; "
                         "such a run is not analysed")
    prov = json.loads((d / provenance.FILE).read_text())
    rid, seed = chosen["run"], chosen["seed"]
    if rid != config.run_id(cfg, seed) or chosen["arm"] != cfg.arm:
        raise ValueError(f"{d}: chosen.json names run {rid!r} (arm {chosen['arm']!r}), but config.yaml with seed "
                         f"{seed} gives {config.run_id(cfg, seed)!r}")
    ledger = Ledger(d / "ledger.jsonl")
    steps = sorted(r["step"] for r in ledger.records() if "artifact" in r and r["run"] == rid)
    n = budget(cfg)
    if steps != list(range(n)):
        raise ValueError(f"{d}: run {rid} has {len(steps)} trials with steps {steps}, but its {cfg.arm} arm's budget is "
                         f"{n} trials with steps 0..{n - 1} once each; a run like this was not made by one invocation "
                         "of one arm and is not analysed")
    trials = sorted((t for t in ledger.trials().values() if t["run"] == rid), key=lambda t: t["step"])
    return Run(d, cfg, rid, seed, chosen["id"], trials, ledger.count(rid), prov)


def budget(cfg: config.RunConfig) -> int:
    """The number of trials an arm generates: 1 for frozen, `max_steps + 1` for refine, `n_candidates` for best-of-N."""
    return {"frozen": 1, "refine": cfg.budget.max_steps + 1, "bestofn": cfg.budget.n_candidates}[cfg.arm]


def load_cell(run_dirs) -> list[Run]:
    """The runs of one finished cell, in the order given; raises ValueError for anything else (see the module)."""
    dirs = [Path(d) for d in run_dirs]
    if not dirs: raise ValueError("no run directories given")
    unfinished = [str(d) for d in dirs if not (d / "done").is_file()]
    if unfinished: raise ValueError(f"these runs did not finish (no `done` file): {unfinished}")
    runs = [_load_run(d) for d in dirs]
    ref = _flat(config.to_dict(runs[0].cfg))
    for r in runs[1:]:
        cur = _flat(config.to_dict(r.cfg))
        diff = sorted(k for k in ref.keys() | cur.keys() if k not in ("arm", "seeds") and ref.get(k) != cur.get(k))
        if diff:
            raise ValueError(f"{runs[0].path} and {r.path} belong to different cells: their configs differ in {diff}")
        prov = sorted(k for k in provenance.COMPARED if runs[0].provenance.get(k) != r.provenance.get(k))
        if prov:
            raise ValueError(f"{runs[0].path} and {r.path} belong to different cells: their {provenance.FILE} differ "
                             f"in {prov}")
    seen: dict[tuple[str, int], Path] = {}
    for r in runs:
        if (r.arm, r.seed) in seen:
            raise ValueError(f"{seen[r.arm, r.seed]} and {r.path} are both the {r.arm} arm with seed {r.seed}")
        seen[r.arm, r.seed] = r.path
    return runs


def _value(rec: dict | None, window: str, metric: str) -> float:
    s = rec["scores"].get(window) if rec else None
    v = s.get(metric) if s and s.get("ok") else None
    return float(v) if v is not None and math.isfinite(v) else math.nan


def run_curve(run: Run, window: str = "heldout", metric: str = "ic") -> list[float]:
    """`scores[window][metric]` of the candidate `run` holds after each step `0..max_steps` (NaN when none); for the
    frozen arm, its one proposal's value on every step."""
    steps = range(run.cfg.budget.max_steps + 1)
    held = [_value(_best([t for t in run.trials if t["step"] <= k]), window, metric) for k in steps]
    return [held[0]] * len(steps) if run.arm == "frozen" else held


def interval(values) -> tuple[float, float, float, int]:
    """(mean, lo, hi, n) over the non-NaN values: the 95% t interval, with NaN bounds when n < 2."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0: return math.nan, math.nan, math.nan, 0
    m = float(x.mean())
    if n < 2: return m, math.nan, math.nan, n
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return m, m - h, m + h, n


@dataclass(frozen=True)
class Paired:
    """A per-seed difference between two steps: its mean, 95% t interval, the seeds it rests on and those left out."""
    mean: float
    lo: float
    hi: float
    n: int
    left_out: int

    @property
    def half_width(self) -> float: return (self.hi - self.lo) / 2


def paired_difference(per_run, a: int, b: int) -> Paired:
    """`value(a) − value(b)` per run, over the runs with a value at both steps (`per_run`: one `run_curve` per run),
    with the `interval` of those differences; runs missing either value are counted in `left_out`."""
    d = [v[a] - v[b] for v in per_run if not (math.isnan(v[a]) or math.isnan(v[b]))]
    return Paired(*interval(d), left_out=len(per_run) - len(d))


def _arms(runs: list[Run]) -> list[str]:
    return [a for a in config.ARMS if any(r.arm == a for r in runs)]


def curve(run_dirs, window: str = "heldout", metric: str = "ic") -> pd.DataFrame:
    """Columns `arm, step, mean, lo, hi, n`: per arm and step, the mean over seeds of `scores[window][metric]` of the
    candidate each run holds after that step (selected on feedback ic), its 95% t interval and the number of seeds
    with a value."""
    runs = load_cell(run_dirs)
    out = []
    for arm in _arms(runs):
        per_run = np.array([run_curve(r, window, metric) for r in runs if r.arm == arm])
        for k in range(per_run.shape[1]):
            out.append((arm, k, *interval(per_run[:, k])))
    return pd.DataFrame(out, columns=CURVE_COLUMNS)


def _feedback_series(rec: dict) -> pd.Series:
    s = rec["scores"]["feedback"]
    return pd.Series(np.asarray(s["daily_ls"], dtype=float), index=s["dates"])


def _sharpe(x: np.ndarray) -> float:
    """Per-period (not annualised) Sharpe ratio, `ddof=1`; NaN for fewer than two points or no spread."""
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else math.nan
    return float(np.mean(x)) / sd if sd > 0 else math.nan


def _dsr(run: Run, valid: list[dict]) -> float:
    if run.chosen is None: return math.nan
    by_id = {t["id"]: t for t in run.trials}
    if run.chosen not in by_id: raise ValueError(f"{run.path}: chosen trial {run.chosen} is not in the ledger")
    x = _feedback_series(by_id[run.chosen]).to_numpy()
    srs = [_sharpe(_feedback_series(t).to_numpy()) for t in valid]
    var = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    return deflated_sharpe(_sharpe(x), run.n_trials, var, len(x), float(stats.skew(x)),
                           float(stats.kurtosis(x, fisher=False)))


def _pbo(valid: list[dict]) -> float:
    if len(valid) < 2: return math.nan
    R = pd.concat([_feedback_series(t) for t in valid], axis=1, join="inner", ignore_index=True).sort_index()
    return pbo_cscv(R.to_numpy(), S=PBO_BLOCKS)["pbo"] if len(R) >= PBO_BLOCKS else math.nan


def selection_stats(run_dirs) -> pd.DataFrame:
    """One row per run: `run, arm, seed, n_trials, n_valid, dsr_of_chosen, pbo`.

    Both statistics are computed on the `feedback` window, because that is the window the selection was made on.
    `n_trials` counts every candidate, invalid ones included; `n_valid` the valid ones with an ok feedback score.
    `dsr_of_chosen` is the deflated Sharpe ratio of the chosen trial's per-period feedback long-short Sharpe, deflated
    for `n_trials` trials with the variance of the `n_valid` trials' Sharpes (0 for fewer than two), the chosen
    series' length, sample skewness and non-excess kurtosis (the moment estimators, `scipy.stats` defaults); NaN when
    nothing was chosen. `pbo` is CSCV with 8 blocks over the `n_valid` trials' feedback long-short series joined on
    their common dates; NaN for fewer than two trials or fewer than 8 common days."""
    rows = []
    for r in load_cell(run_dirs):
        valid = [t for t in r.trials if _eligible(t)]
        rows.append((r.run, r.arm, r.seed, r.n_trials, len(valid), _dsr(r, valid), _pbo(valid)))
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS)
