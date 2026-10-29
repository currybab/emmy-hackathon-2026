"""Elementwise f32 operations: the IR's op name -> one PTX instruction.

add/sub/mul carry no rounding modifier on purpose: that leaves ptxas free to contract a
multiply feeding an add into one `fma` - one instruction instead of two, rounded once instead
of twice. It is what nvcc does by default, so Emmy's reference kernels behave the same way.
div/sqrt/rcp must name a rounding mode for f32; `.rn` is the IEEE-exact one.

PTX has no natural exponential, only ex2 (2^x), so exp is two instructions:
e^x = 2^(x * log2(e)). The destination register doubles as the temporary.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from ptxc.builder import PtxBuilder, f32_literal
from ptxc.errors import CompileError

_ZERO = f32_literal(0.0)
_LOG2_E = f32_literal(1 / math.log(2))

# op name -> (arity, instruction template(s) over {d} destination and {a}, {b} sources)
_F32_OPS: dict[str, tuple[int, str | tuple[str, ...]]] = {
    "add": (2, "add.f32 {d}, {a}, {b}"),
    "subtract": (2, "sub.f32 {d}, {a}, {b}"),
    "multiply": (2, "mul.f32 {d}, {a}, {b}"),
    "divide": (2, "div.rn.f32 {d}, {a}, {b}"),
    "maximum": (2, "max.f32 {d}, {a}, {b}"),
    "relu": (1, "max.f32 {d}, {a}, " + _ZERO),
    "copy": (1, "mov.f32 {d}, {a}"),
    "reciprocal": (1, "rcp.rn.f32 {d}, {a}"),
    "sqrt": (1, "sqrt.rn.f32 {d}, {a}"),
    "exp": (1, ("mul.f32 {d}, {a}, " + _LOG2_E, "ex2.approx.f32 {d}, {d}")),
}


def emit_elementwise(b: PtxBuilder, op: str, dst: str, args: Sequence[str]) -> None:
    """Emit `dst = op(*args)` for f32 registers."""
    if op not in _F32_OPS:
        raise CompileError(f"elementwise op {op!r} is not supported")
    arity, template = _F32_OPS[op]
    if len(args) != arity:
        raise CompileError(f"elementwise op {op!r} takes {arity} operand(s), got {len(args)}")
    operands = dict(zip("ab", args))
    for instruction in (template,) if isinstance(template, str) else template:
        b.emit(instruction.format(d=dst, **operands))
