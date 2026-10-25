"""Integer index expressions (Var / Literal / BinaryExpr trees) -> s32 values.

An `IntValue` is either a Python int, known at compile time, or the name of a b32 register
holding an s32. Keeping constants as ints lets whole subtrees fold away: the IR's
`(a0 * 2) + 0` costs one multiply, and `mul_c1[0]` costs no index arithmetic at all.
"""
from __future__ import annotations

from collections.abc import Mapping

from ptxc.builder import PtxBuilder
from ptxc.errors import CompileError

IntValue = int | str

_INSTRUCTION = {
    "+": "add.s32",
    "-": "sub.s32",
    "*": "mul.lo.s32",
    "/": "div.s32",
    "%": "rem.s32",
}
_COMMUTATIVE = {"+", "*"}


def _c_div(a: int, b: int) -> int:
    """C integer division: truncates toward zero (Python's // floors)."""
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _c_rem(a: int, b: int) -> int:
    return a - b * _c_div(a, b)


_FOLD = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": _c_div,
    "%": _c_rem,
}


def _simplify(op: str, left: IntValue, right: IntValue) -> IntValue | None:
    """Algebraic identities with one constant side; None when none applies."""
    if op == "+" and left == 0:
        return right
    if op in ("+", "-") and right == 0:
        return left
    if op == "*" and (left == 0 or right == 0):
        return 0
    if op == "*" and left == 1:
        return right
    if op in ("*", "/") and right == 1:
        return left
    return None


def int_binop(b: PtxBuilder, op: str, left: IntValue, right: IntValue) -> IntValue:
    """`left op right` with C int semantics, folded when possible, else one instruction."""
    if op not in _INSTRUCTION:
        raise CompileError(f"integer operator {op!r} is not supported")
    if isinstance(left, int) and isinstance(right, int):
        return _FOLD[op](left, right)
    simplified = _simplify(op, left, right)
    if simplified is not None:
        return simplified
    if isinstance(left, int):
        # PTX takes an immediate only as the last source operand
        if op in _COMMUTATIVE:
            left, right = right, left
        else:
            left = materialize(b, left)
    dst = b.reg("b32")
    b.emit(f"{_INSTRUCTION[op]} {dst}, {left}, {right}")
    return dst


def materialize(b: PtxBuilder, value: IntValue) -> str:
    """Force a value into a register."""
    if isinstance(value, str):
        return value
    dst = b.reg("b32")
    b.emit(f"mov.s32 {dst}, {value}")
    return dst


# comparison -> (setp condition, the same comparison with its operands swapped)
_COMPARISONS = {
    "==": ("eq", "=="),
    "!=": ("ne", "!="),
    "<": ("lt", ">"),
    "<=": ("le", ">="),
    ">": ("gt", "<"),
    ">=": ("ge", "<="),
}
_COMPARE_FOLD = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}

Predicate = bool | str  # known at compile time, or a pred register


def compare(b: PtxBuilder, op: str, left: IntValue, right: IntValue) -> Predicate:
    """`left op right` on s32 values: a Python bool when both are constants, else one setp."""
    if op not in _COMPARISONS:
        raise CompileError(f"comparison {op!r} is not supported")
    if isinstance(left, int) and isinstance(right, int):
        return _COMPARE_FOLD[op](left, right)
    if isinstance(left, int):
        op = _COMPARISONS[op][1]  # 5 < x  ->  x > 5
        left, right = right, left
    dst = b.reg("pred")
    b.emit(f"setp.{_COMPARISONS[op][0]}.s32 {dst}, {left}, {right}")
    return dst


def lower_predicate(expr: object, b: PtxBuilder, env: Mapping[str, IntValue]) -> Predicate:
    """Compile a Cond predicate such as `lane == 0`."""
    if type(expr).__name__ != "BinaryExpr" or expr.op not in _COMPARISONS:
        raise CompileError(f"predicate {expr!r} is not a supported comparison")
    left = lower_int_expr(expr.left, b, env)
    right = lower_int_expr(expr.right, b, env)
    return compare(b, expr.op, left, right)


def lower_int_expr(expr: object, b: PtxBuilder, env: Mapping[str, IntValue]) -> IntValue:
    """Compile an integer expression tree; `env` maps axis and index names to values."""
    kind = type(expr).__name__
    if kind == "Literal":
        if not isinstance(expr.value, int) or isinstance(expr.value, bool):
            raise CompileError(f"expected an integer literal in an index, got {expr.value!r}")
        return expr.value
    if kind == "Var":
        if expr.name not in env:
            raise CompileError(f"index variable {expr.name!r} is not defined here")
        return env[expr.name]
    if kind == "BinaryExpr":
        left = lower_int_expr(expr.left, b, env)
        right = lower_int_expr(expr.right, b, env)
        return int_binop(b, expr.op, left, right)
    raise CompileError(f"index expression {kind} is not supported")
