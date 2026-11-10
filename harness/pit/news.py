"""The FNSPID news index as a point-in-time source: `NewsIndex.query(symbols, start_utc, end_utc)` returns the
articles with start_utc <= published < end_utc (the end is exclusive, so an article stamped exactly at the
cutoff is never returned). A `published` column without a timezone is read as UTC. Only the `year=` partitions
the range touches are opened, and the symbol and time filters are applied while reading, so a year of news is
never held whole in memory. Layout: data/fnspid/index/year=YYYY/*.parquet."""
from __future__ import annotations

import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLUMNS = ["published", "stock_symbol", "article_title", "publisher", "url"]
DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "data" / "fnspid" / "index"

class NewsIndex:
    def __init__(self, root=DEFAULT_ROOT): self.root = pathlib.Path(root)

    def query(self, symbols: list[str], start_utc, end_utc) -> pd.DataFrame:
        start, end = pd.Timestamp(start_utc), pd.Timestamp(end_utc)
        if start.tzinfo is None or end.tzinfo is None: raise ValueError("start_utc and end_utc must be timezone-aware")
        start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
        parts = []
        for year in range(start.year, end.year + 1):
            for f in sorted((self.root / f"year={year}").glob("*.parquet")):
                t = self._read(f, list(symbols), start, end)
                if t.num_rows: parts.append(t)
        if not parts: return self._empty()
        df = pa.concat_tables(parts).to_pandas()
        df["published"] = pd.to_datetime(df["published"], utc=True).astype("datetime64[us, UTC]")
        return df[COLUMNS].sort_values("published", kind="stable").reset_index(drop=True)

    @staticmethod
    def _read(f, symbols, start, end) -> pa.Table:
        schema = pq.read_schema(f)
        if not pa.types.is_string(schema.field("stock_symbol").type) and not pa.types.is_large_string(schema.field("stock_symbol").type):
            return pa.table({}, schema=pa.schema([]))      # an all-null symbol column (stored as double) matches no symbol
        tz = schema.field("published").type.tz
        lo, hi = (start, end) if tz else (start.tz_localize(None), end.tz_localize(None))   # no tz in the file means UTC
        ts = pa.timestamp("us", tz=tz)
        flt = (pc.field("stock_symbol").isin(symbols)) & (pc.field("published") >= pa.scalar(lo, ts)) & (pc.field("published") < pa.scalar(hi, ts))
        t = pq.read_table(f, columns=COLUMNS, filters=flt)
        for c in COLUMNS[1:]: t = t.set_column(t.schema.get_field_index(c), c, t[c].cast(pa.large_string()))   # all-null columns are stored as double
        if not tz: t = t.set_column(t.schema.get_field_index("published"), "published", pc.assume_timezone(t["published"], "UTC"))
        return t

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame({"published": pd.Series([], dtype="datetime64[us, UTC]"), **{c: pd.Series([], dtype=object) for c in COLUMNS[1:]}})
