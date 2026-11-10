import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from harness.pit.news import NewsIndex
from harness.pit.registry import LookAheadError, Registry
from harness.pit.tools import make_registry

COLS = ["published", "stock_symbol", "article_title", "publisher", "url"]

def _write(root, year, name, rows, tz):
    """rows: (symbol, 'YYYY-MM-DD HH:MM:SS'); tz None writes a timestamp column without a timezone"""
    d = pathlib.Path(root) / f"year={year}"; d.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime([r[1] for r in rows])
    if tz: ts = ts.tz_localize(tz)
    pq.write_table(pa.table({"article_title": [f"t-{r[1]}" for r in rows], "stock_symbol": [r[0] for r in rows],
                             "url": ["u"] * len(rows), "publisher": ["p"] * len(rows), "article": [None] * len(rows),
                             "published": pa.array(ts, pa.timestamp("us", tz=tz))}), d / name)

@pytest.fixture
def synth(tmp_path):
    root = tmp_path / "index"
    _write(root, 2019, "naive.parquet", [("AAPL", "2019-03-04 12:00:00"), ("AAPL", "2019-03-05 00:00:00"),
                                         ("AAPL", "2019-03-05 05:59:59"), ("AAPL", "2019-03-05 06:00:00"),
                                         ("AAPL", "2019-03-05 06:00:01"), ("MSFT", "2019-03-04 13:00:00")], None)
    _write(root, 2019, "utc.parquet", [("AAPL", "2019-03-04 23:59:59"), ("AAPL", "2019-03-06 01:00:00"),
                                       ("AAPL", "2019-03-01 00:00:00")], "UTC")
    return root

def _reg(tmp_path, root): return make_registry("cn", tmp_path / "calls.jsonl", news_root=root)

def test_naive_published_is_utc(synth):
    df = NewsIndex(root=synth).query(["AAPL"], pd.Timestamp("2019-03-04", tz="UTC"), pd.Timestamp("2019-03-05", tz="UTC"))
    assert list(df["published"]) == [pd.Timestamp("2019-03-04 12:00", tz="UTC"), pd.Timestamp("2019-03-04 23:59:59", tz="UTC")]
    assert str(df["published"].dt.tz) == "UTC" and list(df.columns) == COLS

def test_news_tool_cutoff_exclusive(synth, tmp_path):
    reg = _reg(tmp_path, synth)
    df = reg.call("news", "2019-03-05", symbols=["AAPL"], days=3)
    assert df["published"].max() == pd.Timestamp("2019-03-04 23:59:59", tz="UTC")   # the 00:00:00 row is excluded
    assert str(df["published"].dt.tz) == "UTC" and df["published"].is_monotonic_increasing
    assert pd.Timestamp("2019-03-04 12:00", tz="UTC") in set(df["published"])         # naive row counted as UTC
    assert list(df.columns) == COLS and set(df["stock_symbol"]) == {"AAPL"}

def test_news_tool_cutoff_hour(synth, tmp_path):
    df = _reg(tmp_path, synth).call("news", "2019-03-05", symbols=["AAPL"], days=1, cutoff_hour_utc=6)
    cut = pd.Timestamp("2019-03-05 06:00", tz="UTC")
    assert df["published"].max() == pd.Timestamp("2019-03-05 05:59:59", tz="UTC") and (df["published"] < cut).all()
    assert pd.Timestamp("2019-03-05 00:00", tz="UTC") in set(df["published"])
    with pytest.raises(ValueError): _reg(tmp_path, synth).call("news", "2019-03-05", symbols=["AAPL"], cutoff_hour_utc=25)

def test_news_reads_only_touched_years(synth):
    bad = synth / "year=2018"; bad.mkdir(); (bad / "x.parquet").write_text("not parquet")
    df = NewsIndex(root=synth).query(["AAPL"], pd.Timestamp("2019-03-01", tz="UTC"), pd.Timestamp("2019-03-02", tz="UTC"))
    assert len(df) == 1
    with pytest.raises((OSError, pa.ArrowException)):
        NewsIndex(root=synth).query(["AAPL"], pd.Timestamp("2018-12-30", tz="UTC"), pd.Timestamp("2019-01-02", tz="UTC"))

def test_news_empty_keeps_schema(synth):
    df = NewsIndex(root=synth).query(["ZZZZ"], pd.Timestamp("2019-03-01", tz="UTC"), pd.Timestamp("2019-03-09", tz="UTC"))
    assert df.empty and list(df.columns) == COLS and str(df["published"].dt.tz) == "UTC"

def test_prices_rejects_non_bare_fields(tmp_path):
    reg = make_registry("cn", tmp_path / "calls.jsonl")
    for bad in ["Ref($close, -1)", "$close + 1", "close", "$Close", "$close, $open", ""]:
        with pytest.raises(ValueError): reg.call("prices", "2022-01-04", instruments=["SH600000"], fields=[bad])
    with pytest.raises(ValueError): reg.call("prices", "2022-01-04", instruments=["SH600000"], fields=[])

def test_prices_leak_is_refused(tmp_path, monkeypatch):
    reg = make_registry("cn", tmp_path / "calls.jsonl")
    idx = pd.MultiIndex.from_product([["SH600000"], pd.to_datetime(["2022-01-04", "2022-01-05"])], names=["instrument", "datetime"])
    monkeypatch.setattr(reg.tools["prices"], "fn", lambda as_of, **kw: pd.DataFrame({"$close": [1.0, 2.0]}, index=idx))
    with pytest.raises(LookAheadError): reg.call("prices", "2022-01-04", instruments=["SH600000"], fields=["$close"])

def test_registry_returns_extracts_after_checking_frame(tmp_path):
    reg = Registry(tmp_path / "calls.jsonl")
    @reg.register("ids", returns=lambda f: list(f["id"]))
    def ids(as_of, leak=False): return pd.DataFrame({"id": ["a", "b"], "date": ["2022-01-05" if leak else as_of] * 2})
    assert reg.call("ids", "2022-01-04") == ["a", "b"]
    with pytest.raises(LookAheadError): reg.call("ids", "2022-01-04", leak=True)
    assert (tmp_path / "calls.jsonl").read_text().count("\n") == 1               # the refused call is not logged
    reg2 = Registry(tmp_path / "c2.jsonl")
    reg2.register("plain")(lambda as_of: pd.DataFrame({"date": [as_of]}))
    assert isinstance(reg2.call("plain", "2022-01-04"), pd.DataFrame)

@pytest.mark.data
def test_prices_never_after_as_of(cn_registry):
    df = cn_registry.call("prices", "2022-01-02", instruments=["SH600000"], fields=["$close"], lookback_days=20)  # a Sunday
    assert df.index.get_level_values("datetime").max() <= pd.Timestamp("2021-12-31")
    assert len(df) > 0

@pytest.mark.data
def test_news_cutoff_is_exclusive_and_utc(news_registry):
    df = news_registry.call("news", "2019-03-05", symbols=["AAPL", "JPM"], days=3)   # the 2019 index has no AAPL rows; JPM has some
    assert len(df) > 0
    day = news_registry.call("news", "2019-03-06", symbols=["JPM"], days=1)
    assert (day["published"] == pd.Timestamp("2019-03-05", tz="UTC")).any()          # rows stamped exactly at the cutoff exist ...
    assert not (df["published"] >= pd.Timestamp("2019-03-05", tz="UTC")).any()       # ... and none leaked into the as_of 03-05 call
    assert df["published"].max() < pd.Timestamp("2019-03-05", tz="UTC")
    assert str(df["published"].dt.tz) == "UTC"

@pytest.mark.data
def test_universe_is_point_in_time(us_registry):
    assert "AABA" in us_registry.call("universe", "2017-03-01", market="sp500")
    assert "AABA" not in us_registry.call("universe", "2018-03-01", market="sp500")

@pytest.mark.data
def test_regions_alternate(tmp_path):
    if not ((pathlib.Path.home() / ".qlib/qlib_data/cn_data").exists() and (pathlib.Path.home() / ".qlib/qlib_data/us_data").exists()):
        pytest.skip("needs both cn_data and us_data")
    cn, us = (make_registry(r, tmp_path / f"{r}.jsonl") for r in ("cn", "us"))
    for _ in range(2):
        a = cn.call("prices", "2022-01-10", instruments=["SH600000"], fields=["$close"], lookback_days=10)
        b = us.call("prices", "2022-01-10", instruments=["AAPL"], fields=["$close"], lookback_days=10)
        assert len(a) > 0 and len(b) > 0
        assert set(a.index.get_level_values("instrument")) == {"SH600000"} and set(b.index.get_level_values("instrument")) == {"AAPL"}
