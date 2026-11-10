"""Tests for the backtest. The arithmetic lives in `score_frame`, so most tests build a synthetic
(instrument, datetime) frame and need no data; the `data` tests go through `evaluate` and Qlib."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from harness import backtest
from harness.backtest import FactorScore, evaluate, score_dict, score_frame
from harness.pit.tools import qlib_init
from harness.splits import Window

QLIB = backtest.QLIB_DATA

@pytest.fixture
def cn():
    if not (QLIB / "cn_data").exists(): pytest.skip("no CN data")
    qlib_init("cn")

def make_frame(n_days=70, n_names=40, factor=None, label=None, seed=0):
    """Frame indexed (instrument, datetime) with columns [factor, label]; label is random, factor defaults to it."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    names = [f"S{i:03d}" for i in range(n_names)]
    idx = pd.MultiIndex.from_product([names, dates], names=["instrument", "datetime"])
    y = rng.normal(size=len(idx))
    lab = y if label is None else label(y, idx)
    fac = lab.copy() if factor is None else factor(lab, idx)
    return pd.DataFrame({"f": fac, "y": lab}, index=idx)

def test_momentum_factor_scores_on_csi300(cn):
    fs = evaluate("$close / Ref($close, 20) - 1", Window("2019-01-01", "2019-12-31"), "csi300", "cn")
    assert fs.ok and fs.n_days > 200 and abs(fs.ic) < 0.2 and len(fs.daily_ls) == fs.n_days
def test_constant_expression_fails_cleanly(cn):
    fs = evaluate("$close * 0 + 1", Window("2019-01-01", "2019-03-31"), "csi300", "cn")
    assert not fs.ok and "ic" in fs.reason.lower()
def test_score_dict_has_no_series():
    assert "daily_ls" not in score_dict(FactorScore(True, "", 0, 0, 0, 0, 0, 10, [0.0]*10))

@pytest.mark.data
def test_invalid_expression_fails_with_validator_reason(cn):
    fs = evaluate("Ref($close, -1) / $close", Window("2019-01-01", "2019-03-31"), "csi300", "cn")
    assert not fs.ok and "future" in fs.reason and math.isnan(fs.ic) and fs.daily_ls == [] and fs.dates == []

@pytest.mark.data
def test_second_identical_call_does_not_reload(cn, monkeypatch):
    calls = []
    real = backtest._load
    monkeypatch.setattr(backtest, "_load", lambda *a: calls.append(a) or real(*a))
    backtest._MEMO.clear()
    w = Window("2019-01-01", "2019-06-30")
    a = evaluate("$close / Ref($close, 5) - 1", w, "csi300", "cn")
    b = evaluate("$close / Ref($close, 5) - 1", w, "csi300", "cn")
    assert a == b and len(calls) == 1

def test_perfect_factor_has_ic_one_and_positive_long_short():
    fs = score_frame(make_frame())
    assert fs.ok and fs.ic == pytest.approx(1.0) and fs.ls_mean > 0 and fs.n_days == 70
def test_inverted_factor_has_ic_minus_one():
    fs = score_frame(make_frame(factor=lambda y, i: -y))
    assert fs.ok and fs.ic == pytest.approx(-1.0) and fs.ls_mean < 0
def test_ic_is_spearman_not_pearson():
    fs = score_frame(make_frame(factor=lambda y, i: np.exp(10 * y)))
    assert fs.ic == pytest.approx(1.0)
def test_long_short_uses_top_and_bottom_quantile():
    fs = score_frame(make_frame(n_days=60, n_names=50), quantile=0.2)
    f = make_frame(n_days=60, n_names=50)
    day = f.xs(f.index.get_level_values(1)[0], level="datetime").sort_values("f")
    assert fs.daily_ls[0] == pytest.approx(day.y.iloc[-10:].mean() - day.y.iloc[:10].mean())
def test_day_with_fewer_than_30_valid_names_is_dropped():
    def thin(y, idx):
        out = y.copy(); d = idx.get_level_values(1)
        names = idx.get_level_values(0)
        out[(d == d[0]) & (names >= "S010")] = np.nan   # first day: 10 valid names
        return out
    fs = score_frame(make_frame(factor=thin))
    assert fs.ok and fs.n_days == 69 and len(fs.dates) == 69 and "2020-01-01" not in fs.dates
def test_fewer_than_60_days_fails_with_count_in_reason():
    fs = score_frame(make_frame(n_days=45))
    assert not fs.ok and "45" in fs.reason and "60" in fs.reason and fs.n_days == 0
    assert math.isnan(fs.ic) and fs.daily_ls == [] and fs.dates == []
@pytest.mark.parametrize("factor", [lambda y, i: np.ones_like(y), lambda y, i: np.full_like(y, np.nan)])
def test_constant_or_nan_factor_fails_with_ic_reason_even_when_long(factor):
    fs = score_frame(make_frame(n_days=100, factor=factor))
    assert not fs.ok and "IC" in fs.reason and math.isnan(fs.ic) and fs.dates == []
def test_infinite_values_are_missing():
    def with_inf(y, idx):
        out = y.copy(); d = idx.get_level_values(1); names = idx.get_level_values(0)
        out[(d == d[0]) & (names < "S020")] = np.inf    # first day: 20 valid names left
        out[(d == d[1]) & (names < "S005")] = -np.inf   # second day: 35 valid names left, kept
        return out
    fs = score_frame(make_frame(factor=with_inf))
    assert fs.ok and fs.n_days == 69 and fs.ic == pytest.approx(1.0)
def test_static_ranking_has_zero_turnover():
    fs = score_frame(make_frame(factor=lambda y, i: np.repeat(np.arange(40.0), 70)))
    assert fs.ok and fs.turnover == 0.0
def test_rotation_has_known_turnover():
    # 40 names, quantile 0.2 -> top group of 8. The ranking shifts by 4 names every day, so half the group is new.
    def rotating(y, idx):
        return (idx.codes[0].astype(int) + 4 * idx.codes[1].astype(int)) % 40 * 1.0   # codes are int8
    fs = score_frame(make_frame(factor=rotating))
    assert fs.turnover == pytest.approx(0.5)
def test_dates_align_with_daily_ls():
    fs = score_frame(make_frame())
    assert len(fs.dates) == len(fs.daily_ls) == fs.n_days and fs.dates == sorted(fs.dates)
    assert fs.dates[0] == "2020-01-01"
def test_failed_score_dict_is_strict_json():
    fs = score_frame(make_frame(n_days=10))
    d = score_dict(fs)
    assert "daily_ls" not in d and "dates" not in d and d["ic"] is None and d["ok"] is False
    assert json.loads(json.dumps(d, allow_nan=False)) == d
