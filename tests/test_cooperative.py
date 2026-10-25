"""Smem, Sync, WarpShuffle and TreeHalve on hand-built IR."""
import pytest

from irkit import body_text, lit, load, lowerer, smem, sync, tree_halve, var, warp_shuffle, write
from ptxc.errors import CompileError


def test_smem_is_declared_once_at_module_scope_and_addressed_in_shared_space():
    b, lw = lowerer(ints={"warp": "%r9"})
    lw.lower_body([smem("acc_smem", 4), load("v", "x", lit(0)), write("acc_smem", "v", var("warp"))])
    ptx = b.render("k", params=[], max_threads=128)
    assert ".shared .align 4 .b8 acc_smem[16];" in ptx
    text = body_text(b)
    assert "mov.u64 %rd3, acc_smem;" in text
    assert text[-1].startswith("st.shared.f32 [")


def test_smem_of_another_dtype_is_rejected():
    from emmy.compiler.ir.kernel.ir import Smem

    b, lw = lowerer()
    with pytest.raises(CompileError, match="half"):
        lw.lower_body([Smem(name="s", extents=(4,), dtype="half", align=0)])


def test_sync_is_a_cta_barrier():
    b, lw = lowerer()
    lw.lower_body([sync()])
    assert body_text(b) == ["bar.sync 0;"]


def test_warp_shuffle_is_a_five_step_xor_butterfly():
    b, lw = lowerer()
    lw.lower_body([load("acc0", "x", lit(0)), warp_shuffle(32)])
    shuffles = [line for line in body_text(b) if line.startswith("shfl")]
    offsets = [int(line.split(",")[2]) for line in shuffles]
    assert offsets == [16, 8, 4, 2, 1]
    assert all(line.endswith(", 0x1f, 0xffffffff;") for line in shuffles)


def test_each_butterfly_step_combines_with_the_partner_value():
    b, lw = lowerer()
    lw.lower_body([load("acc0", "x", lit(0)), warp_shuffle(2)])
    assert body_text(b)[1:] == [
        "shfl.sync.bfly.b32 %f2, %f1, 1, 0x1f, 0xffffffff;",
        "add.f32 %f3, %f1, %f2;",
    ]


def test_tree_halve_folds_shared_partials_then_broadcasts_slot_zero():
    b, lw = lowerer(ints={"warp": "%r9"})
    lw.lower_body([smem("acc_smem", 4), load("acc0", "x", lit(0)), tree_halve("acc_smem", 4)])
    text = body_text(b)
    # two halving steps (stride 2, then 1), each gated on warp < stride and closed by a barrier
    assert [line for line in text if line.startswith("setp")] == [
        "setp.lt.s32 %p1, %r9, 2;", "setp.lt.s32 %p2, %r9, 1;"]
    assert text.count("bar.sync 0;") == 2
    # afterwards every thread reads the combined value from slot 0
    assert text[-1].startswith("ld.shared.f32") and text[-1].endswith("[%rd3];")


def test_warp_shuffle_longer_than_a_warp_is_rejected():
    b, lw = lowerer()
    with pytest.raises(CompileError, match="64"):
        lw.lower_body([load("acc0", "x", lit(0)), warp_shuffle(64)])
