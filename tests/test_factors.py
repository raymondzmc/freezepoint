"""Tests for expression validation. Stages 1 to 3 run with no data on disk; the dry run is the `data` test."""
import pytest

from harness import factors
from harness.factors import ALLOWED_OPS, FORBIDDEN, LABEL, MAX_LEN, validate
from harness.pit import tools

CN_DATA = tools.data_present("cn")

@pytest.fixture(autouse=True)
def no_data_dir(request, monkeypatch, tmp_path):
    """Point the dry run at an empty directory so these tests never touch Qlib data, except the `data` test."""
    if request.node.get_closest_marker("data") is None: monkeypatch.setattr(tools, "QLIB_DATA", tmp_path)

def test_negative_ref_is_rejected():
    ok, why = validate("Ref($close, -1) / $close"); assert not ok and "future" in why
def test_unknown_operator_is_rejected():
    ok, why = validate("Magic($close, 5)"); assert not ok and "Magic" in why
def test_alpha158_style_expression_passes():
    ok, _ = validate("Mean($close, 5) / $close - 1"); assert ok
def test_label_itself_is_allowed_only_as_label():
    assert not validate(LABEL)[0]
def test_forbidden_regex_and_ops_are_exported():
    assert FORBIDDEN.search("Ref($close, -1)") and "IdxMax" in ALLOWED_OPS and "Idxmax" not in ALLOWED_OPS

@pytest.mark.parametrize("expr", [
    "Ref(Mean($close, 5), -1)",                    # nested parentheses defeat the regex
    "Ref($close, 0-1)",                            # computed offset
    "Delta($close, -1)",                           # series - series.shift(-1)
    "Mean($close, -5)",                            # any rolling window
    "Corr($close, $volume, -10)",                  # pair rolling
    "Slope($close, -3) / $close",
    "If($close > $open, Ref($close, 1), Ref(Mean($high, 3), -2))",   # negative offset inside an If branch
    "Greater($high, Abs(Ref(Std($close, 5), -1)))",
])
def test_look_ahead_forms_are_rejected(expr):
    ok, why = validate(expr); assert not ok and "future" in why

@pytest.mark.parametrize("expr", [
    "__import__('os').system('true')", "$close.__class__", "().__class__", "'abc'", '"abc"', "lambda x: x",
    "[1, 2]", "{1: 2}", "$close[0]", "$close ** 2", "$close // 2", "$close << 1", "$close = 1", "$close; 1",
    "Mean($close, 5) # c", "Mean($close, 1_0)", "$close\x00", "$closé", "Mean($close, 5).mean()", "$close % 2",
    "eval('1')", "Mean(__builtins__, 5)", "$$close", "$close@5min",
])
def test_python_escapes_are_rejected(expr):
    assert not validate(expr)[0]

@pytest.mark.parametrize("expr, word", [("$vwap", "$vwap"), ("$close + $vwap", "$vwap"), ("Mean($close5, 5)", "$close5")])
def test_unknown_field_is_rejected(expr, word):
    ok, why = validate(expr); assert not ok and word in why

@pytest.mark.parametrize("bad", [None, 5, 5.0, b"$close", ["$close"], "", "   ", "x" * (MAX_LEN + 1), "$close" + " " * MAX_LEN])
def test_non_text_empty_and_overlong_are_rejected(bad):
    ok, why = validate(bad); assert not ok and isinstance(why, str) and why

def test_text_at_the_cap_is_accepted():
    assert validate("$close" + " " * (MAX_LEN - 6))[0]

@pytest.mark.parametrize("expr", [
    "$close", "Mean($close, 5) / $close - 1", "($close-$open)/$open", "Corr($close, Log($volume+1), 10)",
    "Slope($close, 5)/$close", "IdxMax($high, 5)/5", "IdxMin($low, 10) / 10", "Greater($high-$low, Abs($high-Ref($close, 1)))",
    "($high-$low)/($close+1e-12)", "If($close > Ref($close, 1), 1, -1)", "If($close >= $open, $volume, 0 - $volume)",
    "EMA($close, 0.5) / $close", "Std($close, 20) / $close", "Rsquare($close, 5)", "Resi($close, 5) / $close",
    "Quantile($close, 5, 0.8) / $close", "Rank($close, 10)", "Cov($close, $volume, 5)", "EMA($close, 5) / $close",
    "WMA($close, 5) / $close", "Mad($close, 5) / $close", "Sum(Greater($close - Ref($close, 1), 0), 5)",
    "Power($close / Ref($close, 1), 2)", "Sign($close - Ref($close, 1))", "Less($open, $close)", "Max($high, 5) / $close",
    "Min($low, 5) / $close", "$close * $factor", "Mean($close, 5) / 2.5e3", "If(($close > $open) & ($volume != 0), 1, 0)", "If(($close > $open) | ($volume == 0), 1, 0)",
])
def test_legitimate_expressions_pass(expr):
    ok, why = validate(expr); assert ok, why

@pytest.mark.parametrize("expr", ["-$close", "If(~($close > $open), 1, 0)", "Mean($close, 5", "Mean $close", "5Mean($close, 5)", "5"])
def test_allowed_tokens_that_qlib_cannot_build_are_rejected(expr):
    ok, why = validate(expr); assert not ok and why

@pytest.mark.parametrize("expr, words", [
    ("Ref($close, 0) / $close", ["Ref", "0", "positive integer"]),
    ("Delta($close, 0)", ["Delta", "0"]),
    ("Mean($close, 0)", ["Mean", "0"]),
    ("Mean($close, 5 - 5)", ["Mean", "0"]),
    ("Mean($close, $volume)", ["Mean", "expression"]),
    ("Ref($close, Mean($close, 5))", ["Ref", "expression"]),
    ("Mean($close, 2.5)", ["Mean", "2.5"]),
    ("Std($close, 1 > 0)", ["Std", "True"]),            # a comparison of numbers is a bool, not an integer
    ("Ref($close, 2 > 1)", ["Ref", "True"]),
    ("EMA($close, 1.5)", ["EMA", "1.5"]),
    ("EMA($close, 0)", ["EMA", "0"]),
    ("EMA($close, 1.0)", ["EMA", "1.0"]),
    ("Corr($close, $volume, 0)", ["Corr", "0"]),
    ("Quantile($close, 0, 0.8)", ["Quantile", "0"]),
])
def test_window_and_offset_must_be_positive_integers(expr, words):
    ok, why = validate(expr)
    assert not ok and all(w in why for w in words), why

@pytest.mark.parametrize("expr", ["EMA($close, 5)", "EMA($close, 0.5)", "EMA($close, 0.05) / $close", "Ref($close, 1)",
                                  "Mean($close, 2 + 3)"])
def test_positive_integer_windows_and_ema_smoothing_factors_pass(expr):
    ok, why = validate(expr); assert ok, why

@pytest.mark.parametrize("expr", ["$close < $open < $high", "If($low <= $close <= $high, 1, 0)", "$close > $open > 0",
                                  "If(($close > $open) & ($open > $low < $high), 1, 0)"])
def test_chained_comparisons_are_rejected(expr):
    ok, why = validate(expr); assert not ok and "chained" in why, why

def test_negative_window_still_says_future():
    for expr in ("Mean($close, -5)", "Ref($close, 0 - 2)", "EMA($close, -0.5)"):
        ok, why = validate(expr); assert not ok and "future" in why, (expr, why)

HOSTILE = ["__import__('os').system('true')", "$close.__class__", "().__class__", "'abc'", "lambda x: x", "[1, 2]",
           "{1: 2}", "$close[0]", "$close ** 2", "$close = 1", "$close; 1", "Mean($close, 5).mean()", "eval('1')",
           "Mean(__builtins__, 5)", "$close if 1 else 0", "not $close", "$close and $open", "$close @ $open"]

@pytest.mark.parametrize("expr", HOSTILE)
def test_hostile_text_never_reaches_the_parse(monkeypatch, expr):
    monkeypatch.setattr(factors, "_parse", lambda e: pytest.fail(f"hostile text reached the parse: {e!r}"))
    assert not validate(expr)[0]

def test_unknown_region_is_rejected():
    assert not validate("$close", region="jp")[0]

def test_wrong_argument_count_comes_back_as_the_parse_error():
    ok, why = validate("Mean($close)"); assert not ok and "argument" in why

def test_validate_never_raises_on_hostile_input():
    for s in ["(" * 500 + "$close" + ")" * 500, "Mean(" * 100 + "$close" + ", 5)" * 100, "$close/0", "5"]:
        ok, why = validate(s); assert isinstance(ok, bool) and isinstance(why, str)
    assert validate("Mean($close, 5)")[0] and not validate("5")[0]

def test_absent_data_dir_skips_the_dry_run(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "QLIB_DATA", tmp_path / "nowhere")
    monkeypatch.setattr(factors, "qlib_init", lambda region: pytest.fail("the dry run must be skipped"))
    assert validate("Mean($close, 5) / $close - 1") == (True, "")
    assert validate("Ref($close, -1)")[0] is False and validate("Magic($close)")[0] is False

@pytest.mark.data
@pytest.mark.skipif(not CN_DATA, reason="no Qlib cn_data")
def test_dry_run_catches_an_expression_that_parses_but_fails_at_load():
    ok, why = validate("Mean(1, 5)")        # a constant has no .load(); parsing cannot see that
    assert not ok and why
    assert validate("Mean($close, 5) / $close - 1")[0]
