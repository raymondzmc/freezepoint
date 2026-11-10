"""The three tools an agent gets, registered with their availability rules: `prices` and `universe` over Qlib,
`news` over the FNSPID index. Guarantees, each enforced by the registry's check or by argument validation:
prices come back through the last trading day <= as_of (never after) and only as bare fields, since Qlib
evaluates a field string as an expression and `Ref($close, -1)` would otherwise return tomorrow's close;
the universe is the membership on as_of; news is everything published strictly before as_of 00:00 UTC plus
`cutoff_hour_utc`, with a timestamp that has no timezone read as UTC. Each tool initialises Qlib for its own
region at call time, so a cn registry and a us registry can be used alternately in one process."""
from __future__ import annotations

import pathlib

import pandas as pd

from harness.pit.news import DEFAULT_ROOT, NewsIndex
from harness.pit.registry import Registry

FIELDS = ("$open", "$high", "$low", "$close", "$volume", "$factor")
_PROVIDERS = {"cn": "cn_data", "us": "us_data"}
_region: str | None = None

def qlib_init(region: str) -> None:
    """Initialise Qlib for `region` ('cn' or 'us'); a no-op if it is already the current region, a re-init otherwise."""
    global _region
    if region not in _PROVIDERS: raise ValueError(f"unknown region {region!r}; expected one of {sorted(_PROVIDERS)}")
    if _region == region: return
    import qlib
    from qlib.constant import REG_CN, REG_US
    qlib.init(provider_uri=str(pathlib.Path.home() / ".qlib" / "qlib_data" / _PROVIDERS[region]), region={"cn": REG_CN, "us": REG_US}[region])
    _region = region

def make_registry(region: str, log_path, news_root=None) -> Registry:
    if region not in _PROVIDERS: raise ValueError(f"unknown region {region!r}; expected one of {sorted(_PROVIDERS)}")
    reg, index = Registry(pathlib.Path(log_path)), NewsIndex(news_root or DEFAULT_ROOT)

    @reg.register("prices", lag_days=0, doc="daily OHLCV through the last trading day <= as_of; fields are bare Qlib fields such as $close")
    def prices(as_of, instruments, fields, lookback_days=250):
        if not isinstance(fields, (list, tuple)) or not fields or any(f not in FIELDS for f in fields):
            raise ValueError(f"fields must be a non-empty list of bare fields from {FIELDS}; got {fields!r}")
        if not isinstance(instruments, (list, tuple)) or not all(isinstance(i, str) for i in instruments):
            raise ValueError("instruments must be a list of instrument names")
        if not isinstance(lookback_days, int) or lookback_days < 0: raise ValueError("lookback_days must be a non-negative integer")
        from qlib.data import D
        qlib_init(region)
        end = pd.Timestamp(as_of)
        return D.features(list(instruments), list(fields), start_time=end - pd.Timedelta(days=lookback_days), end_time=end)

    @reg.register("universe", lag_days=0, value_dependent=False, returns=lambda f: list(f["instrument"]),
                  doc="instruments of `market` (e.g. sp500, csi300) that are members on as_of")
    def universe(as_of, market):
        from qlib.data import D
        qlib_init(region)
        names = D.list_instruments(D.instruments(market=market), start_time=as_of, end_time=as_of, as_list=True)
        return pd.DataFrame({"instrument": sorted(names), "date": as_of})

    @reg.register("news", lag_days=0, date_col="published",
                  doc="articles about `symbols` published in the `days` before as_of 00:00 UTC + cutoff_hour_utc (exclusive)")
    def news(as_of, symbols, days=5, cutoff_hour_utc=0):
        if not isinstance(cutoff_hour_utc, int) or not 0 <= cutoff_hour_utc <= 24: raise ValueError("cutoff_hour_utc must be an integer in 0..24")
        if not isinstance(days, int) or days < 0: raise ValueError("days must be a non-negative integer")
        end = pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(hours=cutoff_hour_utc)
        return index.query(list(symbols), end - pd.Timedelta(days=days), end)

    return reg
