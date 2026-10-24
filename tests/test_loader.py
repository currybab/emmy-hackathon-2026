import json
import shutil
from pathlib import Path

import pytest

from ptxc.errors import CompileError
from ptxc.loader import load_kernel

PROBLEMS = Path(__file__).resolve().parent.parent / "problems"
P02 = PROBLEMS / "p02_scale_shift" / "k_add_pointwise.ir.json"


def test_entry_name_comes_from_the_ir():
    assert load_kernel(P02).name == "k_add_pointwise"


def test_params_follow_meta_arg_order_not_graph_inputs():
    spec = load_kernel(P02)
    assert [p.name for p in spec.params] == ["mul_c1", "add_c1", "x", "add"]
    assert spec.params[2].shape == (4096,)
    assert spec.params[3].is_output


def test_launch_geometry_comes_from_meta():
    spec = load_kernel(P02)
    assert spec.grid == (4, 1, 1)
    assert spec.block == (256, 1, 1)
    assert spec.total_threads == 1024


def test_tile_is_exposed():
    assert type(load_kernel(P02).tile).__name__ == "Tile"


def test_missing_file_is_a_compile_error(tmp_path):
    with pytest.raises(CompileError, match="cannot read"):
        load_kernel(tmp_path / "nope.ir.json")


def test_missing_meta_is_a_compile_error(tmp_path):
    shutil.copy(P02, tmp_path / P02.name)
    with pytest.raises(CompileError, match="meta.json"):
        load_kernel(tmp_path / P02.name)


def test_kernel_absent_from_meta_is_a_compile_error(tmp_path):
    shutil.copy(P02, tmp_path / P02.name)
    meta = json.loads((P02.parent / "meta.json").read_text())
    meta["kernels"][0]["entry"] = "someone_else"
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(CompileError, match="k_add_pointwise"):
        load_kernel(tmp_path / P02.name)
