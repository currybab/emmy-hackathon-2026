import pytest
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var

from ptxc.builder import PtxBuilder
from ptxc.errors import CompileError
from ptxc.expr import int_binop, lower_int_expr


def lit(v: int) -> Literal:
    return Literal(value=v, dtype="int")


def lines(b: PtxBuilder) -> list[str]:
    return [line.strip() for line in b.body_lines()]


def test_literal_is_a_compile_time_int():
    assert lower_int_expr(lit(7), PtxBuilder(), {}) == 7


def test_var_resolves_through_the_environment():
    assert lower_int_expr(Var(name="a0"), PtxBuilder(), {"a0": "%r3"}) == "%r3"


def test_unknown_var_is_a_clear_error():
    with pytest.raises(CompileError, match="a9"):
        lower_int_expr(Var(name="a9"), PtxBuilder(), {})


def test_constant_subtrees_fold_without_emitting_code():
    b = PtxBuilder()
    expr = BinaryExpr(op="+", left=BinaryExpr(op="*", left=lit(3), right=lit(4)), right=lit(5))
    assert lower_int_expr(expr, b, {}) == 17
    assert lines(b) == []


def test_identities_fold_away():
    b = PtxBuilder()
    a0 = b.reg("b32")
    # (a0 * 1) + 0  ->  a0, no instructions
    expr = BinaryExpr(op="+", left=BinaryExpr(op="*", left=Var(name="a0"), right=lit(1)), right=lit(0))
    assert lower_int_expr(expr, b, {"a0": a0}) == a0
    assert lines(b) == []


def test_register_arithmetic_emits_s32_instructions():
    b = PtxBuilder()
    a0 = b.reg("b32")  # %r1
    expr = BinaryExpr(op="+", left=BinaryExpr(op="*", left=Var(name="a0"), right=lit(2)), right=lit(1))
    assert lower_int_expr(expr, b, {"a0": a0}) == "%r3"
    assert lines(b) == ["mul.lo.s32 %r2, %r1, 2;", "add.s32 %r3, %r2, 1;"]


def test_immediate_left_operand_of_commutative_op_is_swapped():
    b = PtxBuilder()
    x = b.reg("b32")
    int_binop(b, "*", 4, x)
    assert lines(b) == ["mul.lo.s32 %r2, %r1, 4;"]


def test_immediate_left_operand_of_subtraction_is_materialized():
    b = PtxBuilder()
    x = b.reg("b32")
    int_binop(b, "-", 10, x)
    assert lines(b) == ["mov.s32 %r2, 10;", "sub.s32 %r3, %r2, %r1;"]


@pytest.mark.parametrize("op, a, b_, expected", [
    ("/", 7, 2, 3), ("/", -7, 2, -3),  # C truncation, not Python floor
    ("%", 7, 2, 1), ("%", -7, 2, -1),
    ("-", 3, 5, -2),
])
def test_folding_matches_c_integer_semantics(op, a, b_, expected):
    assert int_binop(PtxBuilder(), op, a, b_) == expected


def test_unsupported_operator_is_a_clear_error():
    with pytest.raises(CompileError, match="<<"):
        int_binop(PtxBuilder(), "<<", "%r1", 2)
