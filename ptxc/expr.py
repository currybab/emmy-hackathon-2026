"""Integer index expressions (Var / Literal / BinaryExpr / TernaryExpr trees) -> s32 values.

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
# a non-negative dividend and a positive constant divisor: C's truncating division agrees with
# the unsigned one, which is a shift or a multiply-high instead of a signed-division sequence
_UNSIGNED = {"/": "div.u32", "%": "rem.u32"}


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
    """`left op right` with C int semantics: folded when possible, reused when already
    computed, else one instruction."""
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
    key = (op, left, right)
    known = b.known(key)
    if known is not None:
        return known
    unsigned = op in _UNSIGNED and b.is_non_negative(left) and isinstance(right, int) and right > 0
    dst = b.reg("b32")
    b.emit(f"{_UNSIGNED[op] if unsigned else _INSTRUCTION[op]} {dst}, {left}, {right}")
    if unsigned or (op in _COMMUTATIVE and b.is_non_negative(left) and b.is_non_negative(right)):
        b.mark_non_negative(dst)
    b.remember(key, dst)
    return dst


def materialize(b: PtxBuilder, value: IntValue) -> str:
    """Force a value into a register."""
    if isinstance(value, str):
        return value
    dst = b.reg("b32")
    b.emit(f"mov.s32 {dst}, {value}")
    if value >= 0:
        b.mark_non_negative(dst)
    return dst


# ---- affine offsets ---------------------------------------------------------------------
#
# An Affine (variable, constant) is the value variable + constant. Memory offsets are kept in
# this form so the constant ends up in the address immediate `[base+imm]`, where it is free.
# ptxas cannot do this itself: PTX adds wrap, and the index is sign-extended to 64 bits, so
# `4 * (t + 3)` is not provably `4*t + 12`. Emmy's indices never overflow, so we may.

Affine = tuple[IntValue, int]


def as_affine(value: IntValue) -> Affine:
    return (0, value) if isinstance(value, int) else (value, 0)


def affine_value(b: PtxBuilder, x: Affine) -> IntValue:
    """Collapse to a single value (one add when both parts are present)."""
    return int_binop(b, "+", x[0], x[1])


def affine_add(b: PtxBuilder, x: Affine, y: Affine) -> Affine:
    return int_binop(b, "+", x[0], y[0]), x[1] + y[1]


def affine_scale(b: PtxBuilder, x: Affine, k: int) -> Affine:
    return int_binop(b, "*", x[0], k), x[1] * k


def lower_affine(expr: object, b: PtxBuilder, env: Mapping[str, IntValue]) -> Affine:
    """Like lower_int_expr, but constant terms stay apart: (t + 1) * 64 -> (t*64, 64).

    Only + - and multiplication by a constant distribute; / % and selects collapse first.
    """
    kind = type(expr).__name__
    if kind == "Var" and isinstance(env.get(expr.name), tuple):
        return env[expr.name]  # an unrolled copy's loop variable: counter + constant
    if kind in ("Literal", "Var"):
        return as_affine(lower_int_expr(expr, b, env))
    if kind == "BinaryExpr" and expr.op in ("+", "-", "*"):
        left, right = lower_affine(expr.left, b, env), lower_affine(expr.right, b, env)
        if expr.op == "+":
            return affine_add(b, left, right)
        if expr.op == "-":
            return int_binop(b, "-", left[0], right[0]), left[1] - right[1]
        if left[0] == 0:
            return affine_scale(b, right, left[1])
        if right[0] == 0:
            return affine_scale(b, left, right[1])
        return as_affine(int_binop(b, "*", affine_value(b, left), affine_value(b, right)))
    if kind == "BinaryExpr":
        left, right = lower_affine(expr.left, b, env), lower_affine(expr.right, b, env)
        return as_affine(int_binop(b, expr.op, affine_value(b, left), affine_value(b, right)))
    return as_affine(lower_int_expr(expr, b, env))


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


def _select(expr: object, b: PtxBuilder, env: Mapping[str, IntValue]) -> IntValue:
    """`cond ? if_true : if_false`; a condition known at compile time keeps one side."""
    holds = lower_predicate(expr.cond, b, env)
    if holds is True or holds is False:
        return lower_int_expr(expr.if_true if holds else expr.if_false, b, env)
    if_true = lower_int_expr(expr.if_true, b, env)
    if_false = lower_int_expr(expr.if_false, b, env)
    dst = b.reg("b32")
    b.emit(f"selp.s32 {dst}, {if_true}, {if_false}, {holds}")
    return dst


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
        value = env[expr.name]
        return affine_value(b, value) if isinstance(value, tuple) else value
    if kind == "BinaryExpr":
        left = lower_int_expr(expr.left, b, env)
        right = lower_int_expr(expr.right, b, env)
        return int_binop(b, expr.op, left, right)
    if kind == "TernaryExpr":
        return _select(expr, b, env)
    raise CompileError(f"index expression {kind} is not supported")
