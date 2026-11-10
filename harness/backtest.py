"""The verifier: score one expression on one window and one universe.

Guarantees. (1) `evaluate` is the only place a candidate is executed for scoring, and it validates first: nothing
unvalidated reaches `D.features` through it (the label is appended here and reads the future by design). (2) The
arithmetic is a pure function of an (instrument, datetime) frame, `score_frame`, so it is testable without data.
(3) Scores are deterministic and memoised in-process. (4) A failed score has NaN metrics and no series, so a failure
can never be mistaken for a weak result; `score_dict` writes strict JSON (non-finite floats become null). (5) Qlib
failing on a validated expression is that candidate's failed score, never an exception that ends a run; Qlib failing
to start is not, and propagates."""
from __future__ import annotations

import dataclasses
import itertools
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from harness.factors import LABEL, validate
from harness.pit.tools import qlib_init
from harness.splits import Window

QLIB_DATA = pathlib.Path.home() / ".qlib" / "qlib_data"
MIN_NAMES = 30   # valid names a day needs to be kept
MIN_DAYS = 60    # kept days a score needs to be ok
_MEMO: dict = {}

@dataclass
class FactorScore:
    ok: bool
    reason: str
    ic: float
    icir: float
    ls_mean: float
    ls_sharpe: float
    turnover: float
    n_days: int
    daily_ls: list[float]
    dates: list[str] = field(default_factory=list)   # ISO dates of the kept days, aligned with daily_ls

def _failed(reason: str) -> FactorScore:
    nan = float("nan")
    return FactorScore(False, reason, nan, nan, nan, nan, nan, 0, [], [])

def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 and math.isfinite(b) else float("nan")

def _std(x) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")

def _rank_ic(f: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation; NaN when either side is constant."""
    rf, ry = pd.Series(f).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    rf, ry = rf - rf.mean(), ry - ry.mean()
    den = math.sqrt((rf @ rf) * (ry @ ry))
    return float(rf @ ry / den) if den > 0 else float("nan")

def score_frame(df: pd.DataFrame, quantile: float = 0.2) -> FactorScore:
    """Score a frame indexed (instrument, datetime) whose first column is the expression and second the label."""
    f_all, y_all = (df.iloc[:, i].to_numpy(dtype=float) for i in (0, 1))
    f_all, y_all = np.where(np.isfinite(f_all), f_all, np.nan), np.where(np.isfinite(y_all), y_all, np.nan)
    inst = df.index.get_level_values(0).to_numpy().astype(str)
    dates = df.index.get_level_values(1)
    order = np.argsort(dates.to_numpy(), kind="stable")
    cuts = np.flatnonzero(np.diff(dates.to_numpy()[order]) != 0) + 1
    ics, ls, kept, tops = [], [], [], []
    for rows in np.split(order, cuts):
        ok = np.isfinite(f_all[rows]) & np.isfinite(y_all[rows])
        if ok.sum() < MIN_NAMES: continue
        r = rows[ok]
        f, y, names = f_all[r], y_all[r], inst[r]
        ic = _rank_ic(f, y)
        if math.isnan(ic): continue
        k = max(1, math.floor(len(r) * quantile))
        top = np.lexsort((names, -f))[:k]       # highest value first, ties by instrument name
        bottom = np.lexsort((names, f))[:k]
        ics.append(ic); ls.append(float(y[top].mean() - y[bottom].mean()))
        kept.append(str(pd.Timestamp(dates[rows[0]]).date())); tops.append(frozenset(names[top]))
    if not ics: return _failed("no IC: the expression is constant or NaN across the universe")
    if len(ics) < MIN_DAYS: return _failed(f"only {len(ics)} usable days, below the floor of {MIN_DAYS}")
    turn = [len(b - a) / len(b) for a, b in itertools.pairwise(tops)]
    ic_mean, ls_mean = float(np.mean(ics)), float(np.mean(ls))
    return FactorScore(True, "", ic_mean, _ratio(ic_mean, _std(ics)), ls_mean, _ratio(ls_mean, _std(ls)) * math.sqrt(252),
                       float(np.mean(turn)), len(ics), ls, kept)

def _load(expr: str, window: Window, universe: str, region: str) -> pd.DataFrame:
    """One `D.features` call: the expression and the label over the universe's membership through time."""
    from qlib.data import D
    qlib_init(region)
    return D.features(D.instruments(universe), [expr, LABEL], start_time=window.start, end_time=window.end, disk_cache=0)

def _score(expr: str, window: Window, universe: str, region: str, quantile: float) -> FactorScore:
    qlib_init(region)   # outside the try: a data directory Qlib cannot start on is not a property of the candidate
    try:
        df = _load(expr, window, universe, region)
    except Exception as e:  # noqa: BLE001 - whatever Qlib raises on a validated expression is that candidate's failure
        return _failed(f"Qlib raised while computing the expression: {type(e).__name__}: {e}")
    return score_frame(df, quantile)

def evaluate(expr: str, window: Window, universe: str, region: str, quantile: float = 0.2) -> FactorScore:
    """Score `expr` on `window` over `universe`. Validates first; a rejected expression returns a failed score with
    the validator's reason and is never executed. An exception Qlib raises while computing a validated expression
    is a failed score that names it; an exception from initialising Qlib propagates and nothing is memoised.
    Memoised on every argument, failures included."""
    key = (expr, window, universe, region, quantile)
    if key not in _MEMO:
        ok, why = validate(expr, region)
        _MEMO[key] = _score(expr, window, universe, region, quantile) if ok else _failed(why)
    return _MEMO[key]

def score_dict(fs: FactorScore) -> dict:
    """Everything but the series, as strict JSON: non-finite floats become None."""
    d = {k: v for k, v in dataclasses.asdict(fs).items() if k not in ("daily_ls", "dates")}
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}

def score_record(fs: FactorScore) -> dict:
    """What the ledger stores for one window: `score_dict` plus the daily long-short series and its dates."""
    return {**score_dict(fs), "daily_ls": list(fs.daily_ls), "dates": list(fs.dates)}
