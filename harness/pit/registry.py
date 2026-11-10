"""The tool registry for the protocol's second layer.

A tool is a function of (as_of, **args) whose result must be computable from information available on
`as_of`. Each tool declares `lag_days`, the publication delay of its source (prices: 0 after the close;
fundamentals: the filing lag; news: 0 at the publish timestamp), and the registry enforces
    data_date + lag_days <= as_of
on the frame the tool returns, by checking the frame's date index against `as_of` before handing it back.
This is the value-independent fragment of look-ahead-freedom (Fonseca, arXiv 2607.04958): availability
depends on dates, not on values, so it can be checked in linear time. Tools whose availability depends
on values (e.g. "the day a stock entered the index") must declare `value_dependent=True` and supply
their own availability column; the registry only checks the declared column.

Every call is appended to the call log (JSONL) with the tool name, as_of, args and a hash of the
result, so the audit can replay the run and the ledger can attribute each candidate to the data it saw."""
from __future__ import annotations
import datetime as dt, hashlib, json, pathlib, time
from dataclasses import dataclass, field
from typing import Callable
import pandas as pd

class LookAheadError(RuntimeError): ...

@dataclass
class Tool:
    name: str
    fn: Callable
    lag_days: int = 0
    date_col: str = "date"
    value_dependent: bool = False
    availability_col: str | None = None
    doc: str = ""

@dataclass
class Registry:
    log_path: pathlib.Path
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, name, lag_days=0, date_col="date", value_dependent=False, availability_col=None, doc=""):
        def deco(fn):
            if value_dependent and not availability_col:
                raise ValueError(f"{name}: a value-dependent tool must name its availability column")
            self.tools[name] = Tool(name, fn, lag_days, date_col, value_dependent, availability_col, doc or fn.__doc__ or "")
            return fn
        return deco

    def call(self, name, as_of: str, **args):
        tool = self.tools[name]; as_of_d = dt.date.fromisoformat(as_of)
        out = tool.fn(as_of=as_of, **args)
        if isinstance(out, pd.DataFrame) and len(out):
            col = tool.availability_col if tool.value_dependent else tool.date_col
            if col in out.columns: dates = pd.to_datetime(out[col]).dt.date
            elif isinstance(out.index, pd.DatetimeIndex): dates = out.index.date
            elif out.index.nlevels > 1 and "datetime" in out.index.names: dates = out.index.get_level_values("datetime").date
            else: raise LookAheadError(f"{name}: cannot find the date column {col!r} to check availability")
            latest = max(dates); allowed = as_of_d - dt.timedelta(days=0 if tool.value_dependent else tool.lag_days)
            if latest > allowed:
                raise LookAheadError(f"{name}: returns data dated {latest} but as_of {as_of} with lag {tool.lag_days}d allows <= {allowed}")
        self._log(name, as_of, args, out)
        return out

    def _log(self, name, as_of, args, out):
        h = hashlib.sha256(pd.util.hash_pandas_object(out).values.tobytes() if isinstance(out, pd.DataFrame) else repr(out).encode()).hexdigest()[:16]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"t": time.time(), "tool": name, "as_of": as_of, "args": args, "result_sha": h}, default=str) + "\n")

    def manifest(self):
        """what the audit publishes: every tool, its lag and whether it is value-dependent"""
        return {n: {"lag_days": t.lag_days, "value_dependent": t.value_dependent, "availability_col": t.availability_col, "doc": t.doc.strip()}
                for n, t in self.tools.items()}
