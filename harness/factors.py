"""Expression validation: the gate every candidate passes before it is scored.

Three guarantees. (1) No candidate reads the future: a look-ahead pattern, or any operator with a negative window or
offset in the parsed tree, is rejected. (2) The scored factor is what was written: every window or offset argument `N`
in the parsed tree must be a positive integer (`bool` is not an integer here; 0, which Qlib reads as an expanding
window or no shift, is rejected), except that `EMA`'s may also be a smoothing factor strictly between 0 and 1; an
expression-valued `N` is rejected, and so is a chained comparison (`a < b < c`, which Python evaluates as
`(a < b) and (b < c)` and Qlib would silently reduce to `b < c`). (3) Untrusted text never reaches Python's `eval`
unless every token is on an allow-list: Qlib builds expressions with `eval`, so the allow-list runs first and the parse
uses an empty-builtins namespace of Qlib's own operators, with no data loaded. `validate` never raises."""
from __future__ import annotations

import re
from types import SimpleNamespace

from harness.pit.tools import FIELDS, REGIONS, data_present, qlib_init

ALLOWED_OPS = frozenset([
    "Ref", "Mean", "Std", "Max", "Min", "Sum", "Rank", "Corr", "Cov", "Delta", "Slope", "Rsquare", "Resi", "Quantile", "Abs",
    "Log", "Sign", "Power", "If", "Greater", "Less", "EMA", "WMA", "Mad", "IdxMax", "IdxMin"])
ALLOWED_FIELDS = frozenset(FIELDS)
FORBIDDEN = re.compile(r"Ref\s*\([^)]*,\s*-\s*\d")
LABEL = "Ref($close, -2) / Ref($close, -1) - 1"
MAX_LEN = 1000
_DRY_RUN = {"cn": ("SH600000", "2020-01-02", "2020-02-14"), "us": ("AAPL", "2020-01-02", "2020-02-14")}

_TOKEN = re.compile(r"""
    (?P<ws>[ \t\r\n]+)
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<field>\$[A-Za-z0-9_]*)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<sym>\*\*|//|<<|>>|<=|>=|==|!=|[-+*/(),<>&|~])
""", re.VERBOSE)
_SYMBOLS = frozenset(["+", "-", "*", "/", "(", ")", ",", "<", ">", "<=", ">=", "==", "!=", "&", "|", "~"])

def _allow_list(expr) -> str | None:
    """None if every token is allowed, else the reason."""
    if not isinstance(expr, str): return f"expression must be text, got {type(expr).__name__}"
    if not expr.strip(): return "empty expression"
    if len(expr) > MAX_LEN: return f"expression longer than {MAX_LEN} characters"
    pos = 0
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if m is None: return f"character {expr[pos]!r} at position {pos} is not allowed"
        kind, text = m.lastgroup, m.group()
        if kind == "field" and text not in ALLOWED_FIELDS: return f"unknown field {text}"
        if kind == "name" and text not in ALLOWED_OPS: return f"unknown operator {text}"
        if kind == "sym" and text not in _SYMBOLS: return f"symbol {text!r} is not allowed"
        pos = m.end()
    return None

def _namespace() -> SimpleNamespace:
    """Qlib's operators by class name, built the way `register_all_ops` does but kept local: no `qlib.init`, no data
    provider, no global state touched, no logging."""
    from qlib.data.ops import OpsList
    return SimpleNamespace(**{c.__name__: c for c in OpsList})

def _parse(expr: str):
    """Build Qlib's expression tree with Qlib's own rewriting (`parse_field`) and an empty-builtins eval."""
    from qlib.utils import parse_field
    ns = _namespace()
    return eval(parse_field(expr), {"__builtins__": {}}, {"Operators": ns, "Feature": ns.Feature})

def _chained(expr: str) -> str | None:
    """A chained comparison anywhere in the expression, found on Python's syntax tree of Qlib's rewriting (parsed, never
    evaluated); a syntax error is left to the parse stage to report."""
    import ast

    from qlib.utils import parse_field
    try:
        tree = ast.parse(parse_field(expr), mode="eval")
    except SyntaxError:
        return None
    if any(isinstance(n, ast.Compare) and len(n.ops) > 1 for n in ast.walk(tree)):
        return ("a chained comparison such as a < b < c is not supported (Python reads it as (a < b) and (b < c), "
                "which Qlib reduces to b < c); compare two values at a time and combine them with & or |")
    return None

def _bad_window(node) -> str | None:
    """Walk the tree: the first window or offset `N` that is not a positive integer (or, for `EMA`, a float strictly
    between 0 and 1), with what was found; a negative one is a read of the future."""
    from qlib.data.base import Expression
    name = type(node).__name__
    if hasattr(node, "N"):
        n = node.N
        if isinstance(n, Expression):
            return f"{name}'s window/offset is an expression ({n}), not a positive integer"
        if isinstance(n, (int, float)) and not isinstance(n, bool) and n < 0:
            return f"{name} with window/offset {n} reads the future"
        smoothing = name == "EMA" and isinstance(n, float) and 0 < n < 1
        if not (smoothing or (isinstance(n, int) and not isinstance(n, bool) and n > 0)):
            also = " or a smoothing factor strictly between 0 and 1" if name == "EMA" else ""
            return f"{name}'s window/offset is {n!r}, which is not a positive integer{also}"
    for v in vars(node).values():
        if isinstance(v, Expression) and (why := _bad_window(v)): return why
    return None

def _dry_run(expr: str, region: str) -> str | None:
    """Load the expression for one instrument and about 30 days; the error text on failure."""
    inst, start, end = _DRY_RUN[region]
    try:
        from qlib.data import D
        qlib_init(region)
        D.features([inst], [expr], start_time=start, end_time=end, disk_cache=0)
    except Exception as e:  # noqa: BLE001 - any load failure is the verdict
        return f"{type(e).__name__}: {e}"
    return None

def validate(expr: str, region: str = "cn") -> tuple[bool, str]:
    """(True, "") if `expr` may be scored, else (False, reason). Stages: allow-list, look-ahead regex, chained
    comparisons, parse and tree walk (windows and offsets), then a dry load when the region's data is present (`data_present`); with no data the dry run is skipped,
    so a data fault is never reported as the candidate's. Never raises."""
    try:
        if region not in REGIONS: return False, f"unknown region {region!r}"
        if why := _allow_list(expr): return False, why
        if FORBIDDEN.search(expr): return False, "Ref with a negative offset reads the future"
        if why := _chained(expr): return False, why
        try:
            tree = _parse(expr)
        except Exception as e:  # noqa: BLE001 - anything Qlib's eval raises is a parse failure
            return False, f"{type(e).__name__}: {e}"
        from qlib.data.base import Expression
        if not isinstance(tree, Expression): return False, "not an expression over fields"
        if why := _bad_window(tree): return False, why
        if data_present(region) and (why := _dry_run(expr, region)): return False, why
        return True, ""
    except BaseException as e:   # the contract is "never raises"
        if isinstance(e, (KeyboardInterrupt, SystemExit)): raise
        return False, f"{type(e).__name__}: {e}"
