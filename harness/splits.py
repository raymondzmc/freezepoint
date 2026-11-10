"""Time splits for the two papers. A split is a named set of windows an experiment arm may see, plus the
windows it may never see. The loop gets `feedback`; the diagnostic gets `pilot`; the paper reports
`heldout`; `forward` is the committed window that opens after the commitment date.

Conventions: dates are inclusive ISO strings; every window ends at least `embargo` trading days before
the next one starts so that overlapping labels cannot leak (de Prado's purging and embargo)."""
from dataclasses import dataclass, field
import datetime as dt

@dataclass(frozen=True)
class Window:
    start: str
    end: str
    def __post_init__(self):
        a, b = dt.date.fromisoformat(self.start), dt.date.fromisoformat(self.end)
        if a > b: raise ValueError(f"window ends before it starts: {self}")

@dataclass(frozen=True)
class Split:
    name: str
    feedback: Window          # what the loop scores against
    pilot: Window             # a small out-of-window slice the diagnostic may read; never the loop
    heldout: Window           # scored once per frozen version, after the run
    forward: Window | None    # the committed window, if any
    embargo_days: int = 10
    regime: str = "all"       # "all" | "bull" | "bear": a label, assigned from index returns elsewhere
    universe: str = "csi300"  # instruments file name in the Qlib region

    def check(self):
        order = [self.feedback, self.pilot, self.heldout] + ([self.forward] if self.forward else [])
        for a, b in zip(order, order[1:]):
            gap = (dt.date.fromisoformat(b.start) - dt.date.fromisoformat(a.end)).days
            if gap < self.embargo_days: raise ValueError(f"{self.name}: {a} to {b} gap {gap}d < embargo {self.embargo_days}d")
        return self

DEFAULT = Split(name="cn-2018-2023", feedback=Window("2018-01-01", "2021-12-31"), pilot=Window("2022-01-17", "2022-06-30"),
                heldout=Window("2022-07-15", "2023-12-31"), forward=None).check()

SPLITS: dict[str, Split] = {DEFAULT.name: DEFAULT}

def get(name: str) -> Split:
    if name not in SPLITS: raise ValueError(f"unknown split {name!r}; known splits: {sorted(SPLITS)}")
    return SPLITS[name]
