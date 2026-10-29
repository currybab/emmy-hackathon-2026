import pytest

from ptxc.builder import PtxBuilder
from ptxc.errors import CompileError
from ptxc.elementwise import emit_elementwise


def lines(b: PtxBuilder) -> list[str]:
    return [line.strip() for line in b.body_lines()]


@pytest.mark.parametrize("op, args, expected", [
    # no rounding modifier: ptxas may contract mul+add into fma, as nvcc does
    ("add", ["%f1", "%f2"], "add.f32 %f9, %f1, %f2;"),
    ("subtract", ["%f1", "%f2"], "sub.f32 %f9, %f1, %f2;"),
    ("multiply", ["%f1", "%f2"], "mul.f32 %f9, %f1, %f2;"),
    ("divide", ["%f1", "%f2"], "div.rn.f32 %f9, %f1, %f2;"),
    ("maximum", ["%f1", "%f2"], "max.f32 %f9, %f1, %f2;"),
    ("relu", ["%f1"], "max.f32 %f9, %f1, 0f00000000;"),
    ("copy", ["%f1"], "mov.f32 %f9, %f1;"),
    ("reciprocal", ["%f1"], "rcp.rn.f32 %f9, %f1;"),
    ("sqrt", ["%f1"], "sqrt.rn.f32 %f9, %f1;"),
])
def test_single_instruction_ops(op, args, expected):
    b = PtxBuilder()
    emit_elementwise(b, op, "%f9", args)
    assert lines(b) == [expected]


@pytest.mark.parametrize("op, args, expected", [
    ("minimum", ["%f1", "%f2"], "min.f32 %f9, %f1, %f2;"),
    ("rsqrt", ["%f1"], "rsqrt.approx.f32 %f9, %f1;"),
    ("negative", ["%f1"], "neg.f32 %f9, %f1;"),
    ("abs", ["%f1"], "abs.f32 %f9, %f1;"),
    ("square", ["%f1"], "mul.f32 %f9, %f1, %f1;"),
    ("pad", ["%f1"], "mov.f32 %f9, %f1;"),
])
def test_more_single_instruction_ops(op, args, expected):
    b = PtxBuilder()
    emit_elementwise(b, op, "%f9", args)
    assert lines(b) == [expected]


def test_exp_fast_is_cudas_expf_intrinsic():
    # __expf: e^x = 2^(x * log2 e) in two instructions
    b = PtxBuilder()
    emit_elementwise(b, "exp_fast", "%f9", ["%f1"])
    assert lines(b) == ["mul.f32 %f9, %f1, 0f3FB8AA3B;", "ex2.approx.f32 %f9, %f9;"]


def test_exp_is_range_reduced_around_an_integer_power_of_two():
    b = PtxBuilder()
    x, dst = b.reg("f32"), b.reg("f32")
    emit_elementwise(b, "exp", dst, [x])
    text = lines(b)
    # clamp to [-104, 89], where e^x is still 0 / inf in f32; NaN passes through
    assert text[:2] == ["max.NaN.f32 %f3, %f1, 0fC2D00000;", "min.NaN.f32 %f3, %f3, 0f42B20000;"]
    assert any(line.endswith("0f4B400000;") for line in text)  # i = round(x * log2 e) via 1.5 * 2^23
    assert not any(line.startswith("cvt") for line in text)
    assert sum("0f3FB8AA3B" in line for line in text) == 2 and any("0f32A57060" in line for line in text)
    assert sum(line.startswith("ex2.approx.ftz.f32") for line in text) == 1  # 2^r only
    assert sum(line.startswith("shl.b32") for line in text) == 2  # 2^i from exponent bits
    assert text[-1].startswith("mul.f32 %f2, ")


def test_a_comparison_is_one_or_zero():
    b = PtxBuilder()
    emit_elementwise(b, "less", "%f9", ["%f1", "%f2"])
    assert lines(b) == ["setp.lt.f32 %p1, %f1, %f2;", "selp.f32 %f9, 0f3F800000, 0f00000000, %p1;"]


def test_not_equal_is_true_for_nan_as_in_c():
    b = PtxBuilder()
    emit_elementwise(b, "not_equal", "%f9", ["%f1", "%f2"])
    assert lines(b)[0] == "setp.neu.f32 %p1, %f1, %f2;"


def test_where_selects_on_a_nonzero_condition():
    b = PtxBuilder()
    emit_elementwise(b, "where", "%f9", ["%f1", "%f2", "%f3"])
    assert lines(b) == ["setp.neu.f32 %p1, %f1, 0f00000000;", "selp.f32 %f9, %f2, %f3, %p1;"]


def test_a_composite_op_folding_in_place_builds_its_result_aside():
    b = PtxBuilder()
    acc, other, cond = b.reg("f32"), b.reg("f32"), b.reg("f32")
    emit_elementwise(b, "where", acc, [cond, acc, other])
    assert lines(b)[-1] == "mov.f32 %f1, %f4;"


def test_unknown_op_names_the_op():
    with pytest.raises(CompileError, match="frobnicate"):
        emit_elementwise(PtxBuilder(), "frobnicate", "%f9", ["%f1"])


def test_wrong_arity_is_rejected():
    with pytest.raises(CompileError, match="add"):
        emit_elementwise(PtxBuilder(), "add", "%f9", ["%f1"])
