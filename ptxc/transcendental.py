"""Elementwise ops that take several instructions: the transcendental functions and selects.

PTX's special-function unit only knows base 2 and approximations: ex2.approx, lg2.approx,
sin.approx, cos.approx, rcp/rsqrt. Each op here is built from those with the steps that keep
it close to the libm function Emmy's CUDA reference calls (expf, logf, tanhf, erff, ...).

Every function emits `dst = op(*args)` with `dst` distinct from the arguments.
"""
from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence

from ptxc.builder import PtxBuilder, f32_literal


def _f32(value: float) -> float:
    """The nearest f32 to `value`, back as a Python float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _hi_lo(value: float) -> tuple[str, str]:
    """A constant as hi + lo: hi its nearest f32, lo the part hi misses. Two fmas with hi then
    lo give x*value to about 48 bits, where one multiply keeps only 24."""
    return f32_literal(value), f32_literal(value - _f32(value))


_ZERO, _ONE, _TWO = f32_literal(0.0), f32_literal(1.0), f32_literal(2.0)
_LN2 = f32_literal(math.log(2))
_LOG2E_HI, _LOG2E_LO = _hi_lo(1 / math.log(2))
_TWO_PI_HI, _TWO_PI_LO = _hi_lo(2 * math.pi)
_INV_2PI = f32_literal(1 / (2 * math.pi))
# e^x is 0 below this and inf above that in f32; clamping first keeps every later step finite
_EXP_LO, _EXP_HI = f32_literal(-104.0), f32_literal(89.0)
_EXP2_LO, _EXP2_HI = f32_literal(-150.0), f32_literal(129.0)  # the same bounds for 2^t
# 1.5 * 2^23: adding it rounds to an integer kept in the mantissa (see _exp2_of)
_ROUNDER_VALUE = 1.5 * 2**23
_ROUNDER = f32_literal(_ROUNDER_VALUE)
_ROUNDER_BITS = struct.unpack("<I", struct.pack("<f", _ROUNDER_VALUE))[0]

# erf after Abramowitz & Stegun 7.1.26 (absolute error <= 1.5e-7):
#   erf(|x|) = 1 - t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))) * exp(-x^2),  t = 1 / (1 + p|x|)
_ERF_P = f32_literal(0.3275911)
_ERF_A = [f32_literal(a) for a in (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)]


F32_EXPONENT_BIAS, F32_MANTISSA_BITS = 127, 23


def _power_of_two(b: PtxBuilder, exponent: str) -> str:
    """2^n for an s32 n in the normal range, built straight in the f32 exponent field."""
    bits = b.reg("b32")
    b.emit(f"add.s32 {bits}, {exponent}, {F32_EXPONENT_BIAS}")
    b.emit(f"shl.b32 {bits}, {bits}, {F32_MANTISSA_BITS}")
    value = b.reg("f32")
    b.emit(f"mov.b32 {value}, {bits}")
    return value


def _exp2_of(b: PtxBuilder, dst: str, t: str, x: str | None = None) -> None:
    """dst = 2^t for t in [-150, 129], split as 2^r * 2^i with i = round(t) and |r| <= 1/2.

    Rounding without a conversion instruction (those run at a quarter of the ALU rate): adding
    1.5 * 2^23 rounds t to an integer, which lands in the low mantissa bits of the sum, so i is
    one subtraction away both as a float and as an s32.
    2^r is ex2.approx, accurate on that small range. 2^i is exact and needs no special-function
    unit: it is written as 2^h * 2^(i-h) with h = i/2, so both halves are normal floats and
    overflow to inf or underflow towards 0 happens in the final multiplies, as it should.
    When `x` is given, t = x*log2(e) and r is recomputed from x with the hi/lo constant, so
    the rounding of t costs nothing.
    """
    shifted, i, r, part = (b.reg("f32") for _ in range(4))
    b.emit(f"add.f32 {shifted}, {t}, {_ROUNDER}")
    b.emit(f"sub.f32 {i}, {shifted}, {_ROUNDER}")
    if x is None:
        b.emit(f"sub.f32 {r}, {t}, {i}")
    else:
        neg_i = b.reg("f32")
        b.emit(f"neg.f32 {neg_i}, {i}")
        b.emit(f"fma.rn.f32 {r}, {x}, {_LOG2E_HI}, {neg_i}")
        b.emit(f"fma.rn.f32 {r}, {x}, {_LOG2E_LO}, {r}")
    # |r| <= 1/2 is never subnormal and neither is 2^r: .ftz drops the subnormal handling
    b.emit(f"ex2.approx.ftz.f32 {part}, {r}")
    n, h = b.reg("b32"), b.reg("b32")
    b.emit(f"mov.b32 {n}, {shifted}")
    b.emit(f"sub.s32 {n}, {n}, {_ROUNDER_BITS}")
    b.emit(f"shr.s32 {h}, {n}, 1")
    b.emit(f"sub.s32 {n}, {n}, {h}")
    b.emit(f"mul.f32 {part}, {part}, {_power_of_two(b, h)}")
    b.emit(f"mul.f32 {dst}, {part}, {_power_of_two(b, n)}")


def exp(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    """e^x = 2^(x * log2 e), range-reduced; NaN stays NaN through the clamp (.NaN variants)."""
    x, t = b.reg("f32"), b.reg("f32")
    b.emit(f"max.NaN.f32 {x}, {args[0]}, {_EXP_LO}")
    b.emit(f"min.NaN.f32 {x}, {x}, {_EXP_HI}")
    b.emit(f"mul.f32 {t}, {x}, {_LOG2E_HI}")
    _exp2_of(b, dst, t, x)


def exp_fast(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    """CUDA's __expf, what emmy's fast-exp knob asks for: one multiply and ex2.approx."""
    b.emit(f"mul.f32 {dst}, {args[0]}, {_LOG2E_HI}")
    b.emit(f"ex2.approx.f32 {dst}, {dst}")


def log(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    """ln x = log2(x) * ln 2."""
    b.emit(f"lg2.approx.f32 {dst}, {args[0]}")
    b.emit(f"mul.f32 {dst}, {dst}, {_LN2}")


def sigmoid(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    """1 / (1 + e^-x), Emmy's own expansion."""
    neg, e = b.reg("f32"), b.reg("f32")
    b.emit(f"neg.f32 {neg}, {args[0]}")
    exp(b, e, [neg])
    b.emit(f"add.f32 {e}, {e}, {_ONE}")
    b.emit(f"rcp.rn.f32 {dst}, {e}")


def tanh(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    """tanh x = sign(x) * (1 - 2 / (e^(2|x|) + 1)): never overflows, odd by construction."""
    a, e = b.reg("f32"), b.reg("f32")
    b.emit(f"abs.f32 {a}, {args[0]}")
    b.emit(f"add.f32 {a}, {a}, {a}")
    exp(b, e, [a])
    b.emit(f"add.f32 {e}, {e}, {_ONE}")
    b.emit(f"div.rn.f32 {e}, {_TWO}, {e}")
    b.emit(f"sub.f32 {e}, {_ONE}, {e}")
    b.emit(f"copysign.f32 {dst}, {args[0]}, {e}")


def erf(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    """Abramowitz & Stegun 7.1.26 on |x|, then the sign of x (erf is odd)."""
    a, t, poly, g = (b.reg("f32") for _ in range(4))
    b.emit(f"abs.f32 {a}, {args[0]}")
    b.emit(f"fma.rn.f32 {t}, {a}, {_ERF_P}, {_ONE}")
    b.emit(f"rcp.rn.f32 {t}, {t}")
    b.emit(f"mov.f32 {poly}, {_ERF_A[4]}")
    for coefficient in reversed(_ERF_A[:4]):  # Horner
        b.emit(f"fma.rn.f32 {poly}, {poly}, {t}, {coefficient}")
    b.emit(f"mul.f32 {poly}, {poly}, {t}")
    b.emit(f"mul.f32 {g}, {a}, {a}")
    b.emit(f"neg.f32 {g}, {g}")
    exp(b, g, [g])
    b.emit(f"mul.f32 {g}, {poly}, {g}")
    b.emit(f"sub.f32 {g}, {_ONE}, {g}")
    b.emit(f"copysign.f32 {dst}, {args[0]}, {g}")


def _reduce_angle(b: PtxBuilder, x: str) -> str:
    """x - 2*pi*round(x / 2*pi): sin.approx/cos.approx are accurate on [-pi, pi] only."""
    k, r = b.reg("f32"), b.reg("f32")
    b.emit(f"mul.f32 {k}, {x}, {_INV_2PI}")
    b.emit(f"cvt.rni.f32.f32 {k}, {k}")
    b.emit(f"neg.f32 {k}, {k}")
    b.emit(f"fma.rn.f32 {r}, {k}, {_TWO_PI_HI}, {x}")
    b.emit(f"fma.rn.f32 {r}, {k}, {_TWO_PI_LO}, {r}")
    return r


def sin(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    b.emit(f"sin.approx.f32 {dst}, {_reduce_angle(b, args[0])}")


def cos(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    b.emit(f"cos.approx.f32 {dst}, {_reduce_angle(b, args[0])}")


def pow_(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    """x^y = 2^(y * log2|x|), then C's special cases: a negative base keeps its sign for an
    odd integer y and is NaN for a non-integer y; anything to the power 0 is 1."""
    x, y = args
    t, mag, whole, half, frac = (b.reg("f32") for _ in range(5))
    is_int, is_odd, negative, zero_y = (b.reg("pred") for _ in range(4))
    b.emit(f"abs.f32 {t}, {x}")
    b.emit(f"lg2.approx.f32 {t}, {t}")
    b.emit(f"mul.f32 {t}, {t}, {y}")
    b.emit(f"max.NaN.f32 {t}, {t}, {_EXP2_LO}")  # keep 2^t's exponent in range; NaN passes
    b.emit(f"min.NaN.f32 {t}, {t}, {_EXP2_HI}")
    _exp2_of(b, mag, t)
    b.emit(f"cvt.rzi.f32.f32 {whole}, {y}")
    b.emit(f"setp.eq.f32 {is_int}, {whole}, {y}")
    b.emit(f"mul.f32 {half}, {y}, 0f3F000000")  # y / 2
    b.emit(f"cvt.rzi.f32.f32 {frac}, {half}")
    b.emit(f"setp.neu.f32 {is_odd}, {frac}, {half}")  # y/2 is not whole: y is odd
    b.emit(f"and.pred {is_odd}, {is_odd}, {is_int}")
    b.emit(f"setp.lt.f32 {negative}, {x}, {_ZERO}")
    b.emit(f"and.pred {is_odd}, {is_odd}, {negative}")
    b.emit(f"@{is_odd} neg.f32 {mag}, {mag}")
    b.emit(f"setp.lt.and.f32 {negative}, {x}, {_ZERO}, !{is_int}")
    b.emit(f"@{negative} mov.f32 {mag}, 0f7FFFFFFF")  # NaN
    b.emit(f"setp.eq.f32 {zero_y}, {y}, {_ZERO}")
    b.emit(f"selp.f32 {dst}, {_ONE}, {mag}, {zero_y}")


# ---- comparisons and selects: C's bool <-> float conventions ------------------------------

def _to_float(b: PtxBuilder, dst: str, predicate: str) -> None:
    b.emit(f"selp.f32 {dst}, {_ONE}, {_ZERO}, {predicate}")


def _comparison(condition: str) -> Callable[[PtxBuilder, str, Sequence[str]], None]:
    """`a < b` and friends as 1.0 / 0.0. C's != is true for NaN, hence the unordered neu."""
    def lower(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
        p = b.reg("pred")
        b.emit(f"setp.{condition}.f32 {p}, {args[0]}, {args[1]}")
        _to_float(b, dst, p)
    return lower


def _truthy(b: PtxBuilder, value: str) -> str:
    """C's float-to-bool: nonzero (NaN included) is true."""
    p = b.reg("pred")
    b.emit(f"setp.neu.f32 {p}, {value}, {_ZERO}")
    return p


def logical_and(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    p, q = _truthy(b, args[0]), _truthy(b, args[1])
    b.emit(f"and.pred {p}, {p}, {q}")
    _to_float(b, dst, p)


def logical_or(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    p, q = _truthy(b, args[0]), _truthy(b, args[1])
    b.emit(f"or.pred {p}, {p}, {q}")
    _to_float(b, dst, p)


def logical_not(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    p = b.reg("pred")
    b.emit(f"setp.eq.f32 {p}, {args[0]}, {_ZERO}")
    _to_float(b, dst, p)


def where(b: PtxBuilder, dst: str, args: Sequence[str]) -> None:
    b.emit(f"selp.f32 {dst}, {args[1]}, {args[2]}, {_truthy(b, args[0])}")


# op name -> (arity, lowering)
COMPOSITE_OPS: dict[str, tuple[int, Callable[[PtxBuilder, str, Sequence[str]], None]]] = {
    "exp": (1, exp),
    "exp_fast": (1, exp_fast),
    "log": (1, log),
    "sigmoid": (1, sigmoid),
    "tanh": (1, tanh),
    "erf": (1, erf),
    "sin": (1, sin),
    "cos": (1, cos),
    "pow": (2, pow_),
    "equal": (2, _comparison("eq")),
    "not_equal": (2, _comparison("neu")),
    "greater": (2, _comparison("gt")),
    "less": (2, _comparison("lt")),
    "greater_equal": (2, _comparison("ge")),
    "less_equal": (2, _comparison("le")),
    "bitwise_and": (2, logical_and),
    "bitwise_or": (2, logical_or),
    "bitwise_not": (1, logical_not),
    "where": (3, where),
}
