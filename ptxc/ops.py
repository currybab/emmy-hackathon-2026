"""Elementwise f32 operations: the IR's op name -> one PTX instruction.

Rounding is spelled out (`.rn`) on add/sub/mul. Without it ptxas is free to contract a
multiply followed by an add into one `fma`, which rounds once instead of twice and no longer
matches numpy bit for bit - and several problems check with zero tolerance.
"""
from __future__ import annotations

from collections.abc import Sequence

from ptxc.builder import PtxBuilder, f32_literal
from ptxc.errors import CompileError

_ZERO = f32_literal(0.0)

# op name -> (arity, instruction template over {d} destination and {a}, {b} sources)
_F32_OPS: dict[str, tuple[int, str]] = {
    "add": (2, "add.rn.f32 {d}, {a}, {b}"),
    "subtract": (2, "sub.rn.f32 {d}, {a}, {b}"),
    "multiply": (2, "mul.rn.f32 {d}, {a}, {b}"),
    "divide": (2, "div.rn.f32 {d}, {a}, {b}"),
    "maximum": (2, "max.f32 {d}, {a}, {b}"),
    "relu": (1, "max.f32 {d}, {a}, " + _ZERO),
    "copy": (1, "mov.f32 {d}, {a}"),
    "reciprocal": (1, "rcp.rn.f32 {d}, {a}"),
    "sqrt": (1, "sqrt.rn.f32 {d}, {a}"),
}


def emit_elementwise(b: PtxBuilder, op: str, dst: str, args: Sequence[str]) -> None:
    """Emit `dst = op(*args)` for f32 registers."""
    if op not in _F32_OPS:
        raise CompileError(f"elementwise op {op!r} is not supported")
    arity, template = _F32_OPS[op]
    if len(args) != arity:
        raise CompileError(f"elementwise op {op!r} takes {arity} operand(s), got {len(args)}")
    b.emit(template.format(d=dst, **dict(zip("ab", args))))
