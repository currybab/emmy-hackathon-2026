import pytest

from ptxc.builder import PtxBuilder
from ptxc.errors import CompileError
from ptxc.ops import emit_elementwise


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


def test_unknown_op_names_the_op():
    with pytest.raises(CompileError, match="tanh"):
        emit_elementwise(PtxBuilder(), "tanh", "%f9", ["%f1"])


def test_wrong_arity_is_rejected():
    with pytest.raises(CompileError, match="add"):
        emit_elementwise(PtxBuilder(), "add", "%f9", ["%f1"])
