import pytest

from ptxc.builder import PtxBuilder, f32_literal
from ptxc.errors import CompileError


@pytest.mark.parametrize("value, expected", [
    (0.0, "0f00000000"),
    (1.0, "0f3F800000"),
    (-2.0, "0fC0000000"),
    (-1e30, "0fF149F2CA"),
])
def test_f32_literal_spells_exact_ieee_bits(value, expected):
    assert f32_literal(value) == expected


def test_registers_are_numbered_per_class_from_one():
    b = PtxBuilder()
    assert [b.reg("f32"), b.reg("f32"), b.reg("b32"), b.reg("b64"), b.reg("pred")] == \
        ["%f1", "%f2", "%r1", "%rd1", "%p1"]


def test_unknown_register_class_is_rejected():
    with pytest.raises(CompileError):
        PtxBuilder().reg("f16")


def test_labels_are_unique():
    b = PtxBuilder()
    assert b.label("exit") != b.label("exit")


def test_render_declares_only_used_register_classes_with_counts():
    b = PtxBuilder()
    r = b.reg("f32")
    b.emit(f"mov.f32 {r}, {f32_literal(1.0)}")
    ptx = b.render("k", params=["x"], max_threads=128)
    assert ".reg .f32 %f<2>;" in ptx
    assert ".reg .b32" not in ptx
    assert "\tmov.f32 %f1, 0f3F800000;" in ptx


def test_render_module_header_signature_and_launch_bound():
    b = PtxBuilder()
    ptx = b.render("k_add", params=["x0", "add"], max_threads=256)
    assert ptx.startswith(".version 8.0\n.target sm_89\n.address_size 64\n")
    assert ".visible .entry k_add(\n\t.param .u64 param_x0,\n\t.param .u64 param_add\n)" in ptx
    assert ".maxntid 256, 1, 1" in ptx
    assert ptx.rstrip().endswith("}")


def test_globals_are_placed_before_the_entry():
    b = PtxBuilder()
    b.declare_global(".shared .align 4 .b8 acc_smem[16];")
    ptx = b.render("k", params=[], max_threads=32)
    assert ptx.index(".shared") < ptx.index(".visible .entry")


def test_comments_are_forced_to_ascii():
    b = PtxBuilder()
    b.comment("a0 → thread")
    assert b.render("k", params=[], max_threads=32).isascii()
