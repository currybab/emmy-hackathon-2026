import pytest

from irkit import binop, lit, var
from ptxc.builder import PtxBuilder
from ptxc.errors import CompileError
from ptxc.expr import compare, lower_predicate


def lines(b: PtxBuilder) -> list[str]:
    return [line.strip() for line in b.body_lines()]


def test_comparison_sets_a_predicate_register():
    b = PtxBuilder()
    lane = b.reg("b32")
    p = lower_predicate(binop("==", var("lane"), lit(0)), b, {"lane": lane})
    assert p == "%p1"
    assert lines(b) == ["setp.eq.s32 %p1, %r1, 0;"]


@pytest.mark.parametrize("op, mnemonic", [
    ("==", "eq"), ("!=", "ne"), ("<", "lt"), ("<=", "le"), (">", "gt"), (">=", "ge"),
])
def test_every_comparison_operator(op, mnemonic):
    b = PtxBuilder()
    x = b.reg("b32")
    compare(b, op, x, 4)
    assert lines(b) == [f"setp.{mnemonic}.s32 %p1, %r1, 4;"]


def test_immediate_on_the_left_mirrors_the_comparison():
    b = PtxBuilder()
    x = b.reg("b32")
    compare(b, "<", 5, x)  # 5 < x  ==  x > 5
    assert lines(b) == ["setp.gt.s32 %p1, %r1, 5;"]


def test_constant_comparison_folds_to_a_python_bool():
    b = PtxBuilder()
    assert lower_predicate(binop("<", lit(1), lit(2)), b, {}) is True
    assert lower_predicate(binop("==", lit(1), lit(2)), b, {}) is False
    assert lines(b) == []


def test_non_comparison_predicate_is_a_clear_error():
    with pytest.raises(CompileError, match="predicate"):
        lower_predicate(var("lane"), PtxBuilder(), {"lane": "%r1"})
