"""StridedLoop, Accum, Cond and Let on hand-built IR."""
import pytest

from emmy.compiler.dtype import F32
from emmy.compiler.ir.stmt.leaves import Let

from irkit import accum, assign, binop, body_text, cond, let, lit, load, lowerer, strided_loop, var, write
from ptxc.errors import CompileError


def test_strided_loop_is_a_top_tested_loop_over_its_axis():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([strided_loop("a0", 1024, var("t"), lit(128), [load("in0", "x", var("a0"))])])
    text = body_text(b)
    top, done = text[1], text[-1]
    assert text[0] == "mov.s32 %r1, %r9;"  # loop counter starts at `start`
    assert top.endswith(":") and done.endswith(":")
    assert text[2:4] == ["setp.ge.s32 %p1, %r1, 1024;", f"@%p1 bra {done[:-1]};"]
    assert "ld.global.f32 %f1" in " ".join(text)  # the body indexes with the counter
    assert text[-3:-1] == ["add.s32 %r1, %r1, 128;", f"bra {top[:-1]};"]


def test_accumulators_in_the_loop_body_are_seeded_with_their_identity_before_the_loop():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([strided_loop("a0", 1024, var("t"), lit(128), [
        load("in0", "x", var("a0")),
        accum("acc0", "in0", "add"),
        accum("acc1", "in0", "maximum"),
    ])])
    text = body_text(b)
    assert text[:2] == ["mov.f32 %f1, 0f00000000;", "mov.f32 %f2, 0fF149F2CA;"]  # 0 and -1e30
    assert "add.f32 %f1, %f1, %f3;" in text  # acc0 folds in place
    assert "max.f32 %f2, %f2, %f3;" in text


def test_accum_with_a_base_folds_the_value_into_the_base():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([strided_loop("a0", 1024, var("t"), lit(128), [
        load("in0", "x", var("a0")),
        assign("sa", "multiply", "acc1", "in0"),
        accum("acc1", "in0", "add", base="sa"),
    ])])
    assert "add.f32 %f1, %f3, %f2;" in body_text(b)  # acc1 = sa + in0


def test_unseeded_loop_requires_an_existing_accumulator():
    b, lw = lowerer(ints={"t": "%r9"})
    loop = strided_loop("a0", 1024, var("t"), lit(128), [load("in0", "x", var("a0")), accum("acc0", "in0")],
                        seed=False)
    with pytest.raises(CompileError, match="acc0"):
        lw.lower_body([loop])


def test_accumulator_is_visible_after_the_loop():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([
        strided_loop("a0", 1024, var("t"), lit(128), [load("in0", "x", var("a0")), accum("acc0", "in0")]),
        write("out", "acc0", lit(0)),
    ])
    assert body_text(b)[-1] == "st.global.f32 [%rd2], %f1;"


def test_loop_axis_is_scoped_to_the_loop():
    b, lw = lowerer(ints={"t": "%r9"})
    with pytest.raises(CompileError, match="a0"):
        lw.lower_body([
            strided_loop("a0", 1024, var("t"), lit(128), [load("in0", "x", var("a0"))]),
            load("in1", "x", var("a0")),
        ])


def test_cond_branches_around_its_body():
    b, lw = lowerer(ints={"lane": "%r9"})
    lw.lower_body([cond(binop("==", var("lane"), lit(0)), [load("in0", "x", lit(0))])])
    text = body_text(b)
    assert text[0] == "setp.eq.s32 %p1, %r9, 0;"
    assert text[1] == f"@!%p1 bra {text[-1][:-1]};"
    assert text[2] == "ld.global.f32 %f1, [%rd1];"


def test_cond_with_else_takes_exactly_one_branch():
    b, lw = lowerer(ints={"lane": "%r9"})
    lw.lower_body([cond(binop("==", var("lane"), lit(0)),
                        [load("in0", "x", lit(0))],
                        [load("in1", "x", lit(1))])])
    text = body_text(b)
    else_label, end_label = text[4], text[-1]
    assert text[1] == f"@!%p1 bra {else_label[:-1]};"
    assert text[3] == f"bra {end_label[:-1]};"
    assert text[5] == "ld.global.f32 %f2, [%rd1+4];"


def test_constant_true_cond_emits_only_its_body():
    b, lw = lowerer()
    lw.lower_body([cond(binop("<", lit(0), lit(1)), [load("in0", "x", lit(0))], [load("in1", "x", lit(1))])])
    assert body_text(b) == ["ld.global.f32 %f1, [%rd1];"]


def test_let_binds_a_float_literal_to_a_register():
    b, lw = lowerer()
    lw.lower_body([let("one", 1.0)])
    assert body_text(b) == ["mov.f32 %f1, 0f3F800000;"]


def test_later_statements_can_use_a_let_name():
    b, lw = lowerer()
    lw.lower_body([let("one", 1.0), load("in0", "x", lit(0)), assign("y", "multiply", "in0", "one")])
    assert "mul.f32 %f3, %f2, %f1;" in body_text(b)


def test_let_of_a_non_float_value_fails_loudly():
    b, lw = lowerer()
    with pytest.raises(CompileError, match="Let 'i'"):
        lw.lower_body([Let(name="i", value=var("a0"), dtype=F32)])
