"""Expression validation: the gate every candidate passes before it is scored.

Two guarantees. (1) No candidate reads the future: a look-ahead pattern, or any operator with a negative window or
offset in the parsed tree, is rejected. (2) Untrusted text never reaches Python's `eval` unless every token is on an
allow-list: Qlib builds expressions with `eval`, so the allow-list runs first and the parse uses an empty-builtins
namespace of Qlib's own operators, with no data loaded. `validate` never raises."""
from __future__ import annotations

import pathlib
import re
from types import SimpleNamespace

from harness.pit.tools import qlib_init

ALLOWED_OPS = frozenset([
    "Ref", "Mean", "Std", "Max", "Min", "Sum", "Rank", "Corr", "Cov", "Delta", "Slope", "Rsquare", "Resi", "Quantile", "Abs",
    "Log", "Sign", "Power", "If", "Greater", "Less", "EMA", "WMA", "Mad", "IdxMax", "IdxMin"])
ALLOWED_FIELDS = frozenset(["$open", "$high", "$low", "$close", "$volume", "$factor"])
FORBIDDEN = re.compile(r"Ref\s*\([^)]*,\s*-\s*\d")
LABEL = "Ref($close, -2) / Ref($close, -1) - 1"
MAX_LEN = 1000
QLIB_DATA = pathlib.Path.home() / ".qlib" / "qlib_data"
_DIRS = {"cn": "cn_data", "us": "us_data"}
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

def _look_ahead(node) -> str | None:
    """Walk the tree: any child expression, and any `N` (window or offset) that is negative, is a read of the future."""
    from qlib.data.base import Expression
    n = getattr(node, "N", None)
    if isinstance(n, (int, float)) and not isinstance(n, bool) and n < 0:
        return f"{type(node).__name__} with window/offset {n} reads the future"
    for v in vars(node).values():
        if isinstance(v, Expression) and (why := _look_ahead(v)): return why
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
    """(True, "") if `expr` may be scored, else (False, reason). Stages: allow-list, look-ahead regex, parse and tree
    walk, then a dry load when the region's data directory exists. Never raises."""
    try:
        if region not in _DIRS: return False, f"unknown region {region!r}"
        if why := _allow_list(expr): return False, why
        if FORBIDDEN.search(expr): return False, "Ref with a negative offset reads the future"
        try:
            tree = _parse(expr)
        except Exception as e:  # noqa: BLE001 - anything Qlib's eval raises is a parse failure
            return False, f"{type(e).__name__}: {e}"
        from qlib.data.base import Expression
        if not isinstance(tree, Expression): return False, "not an expression over fields"
        if why := _look_ahead(tree): return False, why
        if (QLIB_DATA / _DIRS[region]).exists() and (why := _dry_run(expr, region)): return False, why
        return True, ""
    except BaseException as e:   # the contract is "never raises"
        if isinstance(e, (KeyboardInterrupt, SystemExit)): raise
        return False, f"{type(e).__name__}: {e}"
