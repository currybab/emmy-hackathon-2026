"""Whole-kernel compiles of category 2. Structural checks only; tests/test_remote.py runs them."""
from pathlib import Path

import pytest

from ptxc.compiler import compile_file

PROBLEMS = Path(__file__).resolve().parent.parent / "problems"
REDUCTIONS = ["r01_array_sum", "r02_big_sum", "r03_mean", "r04_row_sum"]


def ir_of(problem: str) -> Path:
    return next((PROBLEMS / problem).glob("*.ir.json"))


@pytest.mark.parametrize("problem", REDUCTIONS)
def test_reduction_kernels_use_the_full_reduction_machinery(problem):
    ptx = compile_file(ir_of(problem))
    assert ptx.isascii()
    assert ptx.count("shfl.sync.bfly.b32") == 5
    assert ".shared .align 4 .b8 acc0_smem[16];" in ptx
    assert "bar.sync 0;" in ptx
    assert "setp.ge.u32" not in ptx  # launch matches the Tile: no guard


def test_lane_and_warp_are_derived_from_the_thread_index():
    ptx = compile_file(ir_of("r01_array_sum"))
    assert "and.b32" in ptx and ", 31;" in ptx
    assert "shr.u32" in ptx and ", 5;" in ptx


def test_pointwise_kernels_do_not_compute_lane_or_warp():
    assert "shr.u32" not in compile_file(ir_of("p00_scalar_add"))


def test_row_reduction_decodes_two_axes():
    ptx = compile_file(ir_of("r04_row_sum"))
    assert "rem.u32" in ptx and "div.u32" in ptx


def test_mean_divides_inside_the_writer_branch():
    assert "div.rn.f32" in compile_file(ir_of("r03_mean"))
