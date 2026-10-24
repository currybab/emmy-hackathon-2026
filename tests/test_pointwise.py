"""Whole-kernel compiles of category 1. Structural checks only; tests/test_remote.py runs them."""
import re
from pathlib import Path

import pytest

from ptxc.compiler import compile_file

PROBLEMS = Path(__file__).resolve().parent.parent / "problems"
POINTWISE = ["p00_scalar_add", "p01_vector_add", "p02_scale_shift", "p03_relu_mul"]


def ir_of(problem: str) -> Path:
    return next((PROBLEMS / problem).glob("*.ir.json"))


@pytest.mark.parametrize("problem", POINTWISE)
def test_compiles_to_ascii_ptx_with_the_ir_entry_name(problem):
    ptx = compile_file(ir_of(problem))
    assert ptx.isascii()
    assert ptx.startswith(".version 8.0")
    assert re.search(r"\.visible \.entry k_\w+_pointwise\(", ptx)


def test_guard_is_emitted_when_threads_outnumber_cells():
    # p00: 256 threads, 64 cells
    ptx = compile_file(ir_of("p00_scalar_add"))
    assert "setp.ge.u32" in ptx and ", 64;" in ptx


def test_guard_is_omitted_when_the_launch_matches_the_tile():
    # p01: 2 x 256 threads, 512 cells
    assert "setp" not in compile_file(ir_of("p01_vector_add"))


def test_grouped_writes_become_vector_stores():
    assert "st.global.v2.f32" in compile_file(ir_of("p00_scalar_add"))
    assert "st.global.v4.f32" in compile_file(ir_of("p02_scale_shift"))


def test_relu_lowers_to_max_with_zero():
    assert "max.f32" in compile_file(ir_of("p03_relu_mul"))


def test_mul_and_add_are_never_contracted():
    ptx = compile_file(ir_of("p02_scale_shift"))
    assert "mul.rn.f32" in ptx and "add.rn.f32" in ptx and "fma" not in ptx
