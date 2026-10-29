"""Elementwise f32 operations: the IR's op name -> one PTX instruction.

add/sub/mul carry no rounding modifier on purpose: that leaves ptxas free to contract a
multiply feeding an add into one `fma` - one instruction instead of two, rounded once instead
of twice. It is what nvcc does by default, so Emmy's reference kernels behave the same way.
div/sqrt/rcp must name a rounding mode for f32; `.rn` is the IEEE-exact one.

Ops that need several instructions (exp, tanh, comparisons, ...) live in transcendental.py.
"""
from __future__ import annotations

from collections.abc import Sequence

from ptxc.builder import PtxBuilder, f32_literal
from ptxc.errors import CompileError
from ptxc.transcendental import COMPOSITE_OPS

_ZERO = f32_literal(0.0)

# op name -> (arity, instruction template over {d} destination and {a}, {b} sources)
_F32_OPS: dict[str, tuple[int, str]] = {
    "add": (2, "add.f32 {d}, {a}, {b}"),
    "subtract": (2, "sub.f32 {d}, {a}, {b}"),
    "multiply": (2, "mul.f32 {d}, {a}, {b}"),
    "divide": (2, "div.rn.f32 {d}, {a}, {b}"),
    "maximum": (2, "max.f32 {d}, {a}, {b}"),
    "minimum": (2, "min.f32 {d}, {a}, {b}"),
    "relu": (1, "max.f32 {d}, {a}, " + _ZERO),
    "copy": (1, "mov.f32 {d}, {a}"),
    "pad": (1, "mov.f32 {d}, {a}"),  # Emmy renders it as its input, like copy
    "reciprocal": (1, "rcp.rn.f32 {d}, {a}"),
    "sqrt": (1, "sqrt.rn.f32 {d}, {a}"),
    "rsqrt": (1, "rsqrt.approx.f32 {d}, {a}"),  # CUDA's rsqrtf is the same approximation
    "negative": (1, "neg.f32 {d}, {a}"),
    "abs": (1, "abs.f32 {d}, {a}"),
    "square": (1, "mul.f32 {d}, {a}, {a}"),
}


def emit_elementwise(b: PtxBuilder, op: str, dst: str, args: Sequence[str]) -> None:
    """Emit `dst = op(*args)` for f32 registers."""
    if op in _F32_OPS:
        arity, template = _F32_OPS[op]
    elif op in COMPOSITE_OPS:
        arity, lower = COMPOSITE_OPS[op]
    else:
        raise CompileError(f"elementwise op {op!r} is not supported")
    if len(args) != arity:
        raise CompileError(f"elementwise op {op!r} takes {arity} operand(s), got {len(args)}")
    if op in _F32_OPS:
        b.emit(template.format(d=dst, **dict(zip("ab", args))))
    elif dst in args:  # an Accum folding in place: build the result aside, then move it
        result = b.reg("f32")
        lower(b, result, args)
        b.emit(f"mov.f32 {dst}, {result}")
    else:
        lower(b, dst, args)
