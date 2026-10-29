"""The statements a tiled matmul adds: Init, Loop, cp.async, ternary indices, atomic writes,
and a shared-memory pool for tiles past the 48 KB static limit."""
import pytest

from irkit import (accum, assign, atomic_add, binop, body_text, cp_async, cp_async_commit, cp_async_wait,
                   init, lit, load, loop, lowerer, smem, ternary, var, write)
from ptxc.errors import CompileError


def test_init_sets_a_register_to_its_starting_value():
    b, lw = lowerer()
    lw.lower_body([init("acc0", 0.0)])
    assert body_text(b) == ["mov.f32 %f1, 0f00000000;"]


def test_loop_runs_its_axis_from_zero_in_steps_of_one():
    b, lw = lowerer()
    lw.lower_body([loop("k", 64, [load("v", "x", var("k"))])])
    text = body_text(b)
    assert text[0] == "mov.s32 %r1, 0;"
    assert "setp.ge.s32 %p1, %r1, 64;" in text
    assert "add.s32 %r1, %r1, 1;" in text


def test_an_unrolled_loop_repeats_its_body_with_the_index_as_a_constant():
    b, lw = lowerer()
    lw.lower_body([init("acc"), loop("k", 3, [load("v", "x", var("k")), accum("acc", "v")], unroll=True)])
    text = body_text(b)
    # k is known in each copy, so every address is the base plus an immediate - no index math
    assert [line for line in text if line.startswith("ld.")] == [
        "ld.global.f32 %f2, [%rd1];", "ld.global.f32 %f3, [%rd1+4];", "ld.global.f32 %f4, [%rd1+8];"]
    assert not any(line.startswith(("setp", "bra")) for line in text)


def test_a_ternary_index_known_at_compile_time_keeps_one_side():
    b, lw = lowerer()
    lw.lower_body([load("v", "x", ternary(binop("<", lit(1), lit(2)), lit(3), lit(5)))])
    assert body_text(b) == ["ld.global.f32 %f1, [%rd1+12];"]


def test_a_runtime_ternary_index_is_a_select():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([load("v", "x", ternary(binop("<", var("t"), lit(512)), var("t"), lit(448)))])
    text = body_text(b)
    assert text[:2] == ["setp.lt.s32 %p1, %r9, 512;", "selp.s32 %r1, %r9, 448, %p1;"]


def test_an_atomic_write_adds_into_global_memory():
    b, lw = lowerer()
    lw.lower_body([load("v", "x", lit(0)), atomic_add("out", "v", lit(2))])
    assert body_text(b)[-1] == "red.global.add.f32 [%rd2+8], %f1;"


def test_cp_async_copies_16_bytes_global_to_shared_through_l2():
    b, lw = lowerer(ints={"t": "%r9"})
    lw.lower_body([smem("tile", 8, 4), cp_async("tile", [lit(1), lit(0)], "x", [var("t")])])
    copy = [line for line in body_text(b) if line.startswith("cp.async")]
    assert copy == ["cp.async.cg.shared.global [%rd3+16], [%rd5], 16;"]


def test_cp_async_smaller_than_16_bytes_goes_through_l1():
    b, lw = lowerer()
    lw.lower_body([smem("tile", 8, 4), cp_async("tile", [lit(0), lit(0)], "x", [lit(0)], nbytes=4)])
    assert body_text(b)[-1] == "cp.async.ca.shared.global [%rd3], [%rd1], 4;"


def test_cp_async_lane_index_adds_the_lanes_row_and_column_to_the_stripe_base():
    b, lw = lowerer()
    stmt = cp_async("tile", [lit(2), lit(0)], "x", [lit(0)], lane=[lit(1), lit(3)])
    lw.lower_body([smem("tile", 8, 4), stmt])
    # (2, 0) + lane (1, 3) on 4 columns: element 2*4 + 1*4 + 3 = 15 -> byte 60
    assert body_text(b)[-1] == "cp.async.cg.shared.global [%rd3+60], [%rd1], 16;"


def test_commit_and_wait_groups():
    b, lw = lowerer()
    lw.lower_body([cp_async_commit(), cp_async_wait(2)])
    assert body_text(b) == ["cp.async.commit_group;", "cp.async.wait_group 2;"]


def test_with_a_dynamic_pool_shared_arrays_are_carved_out_of_it_in_order():
    b, lw = lowerer(smem_pool_bytes=96)
    lw.lower_body([smem("a", 4, 4), smem("b", 2, 4)])
    assert body_text(b) == ["mov.u64 %rd3, smem_pool;", "mov.u64 %rd4, smem_pool;", "add.s64 %rd4, %rd4, 64;"]
    assert any(".extern .shared .align 16 .b8 smem_pool[];" in line for line in b.render("k", [], 32).splitlines())


def test_a_pool_too_small_for_its_arrays_fails_loudly():
    b, lw = lowerer(smem_pool_bytes=32)
    with pytest.raises(CompileError, match="pool"):
        lw.lower_body([smem("a", 4, 4)])
