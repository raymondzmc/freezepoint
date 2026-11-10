"""Selection-aware statistics over the ledger.

deflated_sharpe: Bailey & López de Prado (2014), the probability that the observed Sharpe ratio exceeds
the expected maximum Sharpe ratio of N independent trials with the given variance of trial Sharpes.
The 2026 variants (DSR-L / DSR-EO, López de Prado & Porcu, JPM 53(1)) and the e-value online FDR for
dependent sequential trials go in here next; the ledger already records what they need.

pbo_cscv: Bailey, Borwein, López de Prado & Zhu (2017), the probability of backtest overfitting by
combinatorially symmetric cross-validation over a matrix of trial returns (T x N)."""
from __future__ import annotations
import itertools, math
import numpy as np
from scipy import stats

EULER = 0.5772156649015329

def expected_max_sharpe(n_trials: int, var_trial_sharpe: float) -> float:
    """E[max of N iid Sharpes] under the false strategy theorem, in the same units as the trial Sharpes"""
    if n_trials <= 1: return 0.0
    z = (1 - EULER) * stats.norm.ppf(1 - 1 / n_trials) + EULER * stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_trial_sharpe) * z

def deflated_sharpe(sr: float, n_trials: int, var_trial_sharpe: float, T: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(SR* > 0 | the search), with sr and the trial Sharpes per period, T periods"""
    sr0 = expected_max_sharpe(n_trials, var_trial_sharpe)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))

def pbo_cscv(returns: np.ndarray, S: int = 16) -> dict:
    """returns: T x N matrix of per-period returns of N trials. Splits T into S blocks, forms all C(S, S/2)
    train/test pairings, picks the best in-sample trial, ranks it out-of-sample; PBO is the share of
    pairings where the in-sample best is below the out-of-sample median."""
    T, N = returns.shape
    if S % 2 or S > T: raise ValueError("S must be even and <= T")
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for train_idx in itertools.combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in train_idx]); te = np.concatenate([blocks[i] for i in range(S) if i not in train_idx])
        sr = lambda r: r.mean(0) / (r.std(0, ddof=1) + 1e-12)
        best = int(np.argmax(sr(returns[tr]))); oos = sr(returns[te])
        rank = (stats.rankdata(oos)[best]) / (N + 1)                  # relative rank in (0, 1)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_pairings": int(len(logits)), "mean_logit": float(logits.mean())}
