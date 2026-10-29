"""Grouped block rasterization: the Tile decode that walks G block-rows at a time.

The decode is integer arithmetic only, so a tiny interpreter runs the emitted PTX for every gid
and the test compares the (a, b) it produces with the order the swizzle is meant to give.
"""
from types import SimpleNamespace

import pytest

from irkit import axis
from ptxc.builder import PtxBuilder
from ptxc.compiler import _decode_axes
from ptxc.errors import CompileError

_OPS = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "div": lambda a, b: a // b, "rem": lambda a, b: a % b, "min": min,
}


def run_int_ptx(lines: list[str], regs: dict[str, int]) -> dict[str, int]:
    """Execute straight-line non-negative integer PTX (mov/add/sub/mul/div/rem/min)."""
    regs = dict(regs)
    for line in lines:
        if line.startswith("//"):
            continue
        op, operands = line.rstrip(";").split(None, 1)
        dst, *srcs = [o.strip() for o in operands.split(",")]
        values = [regs[s] if s.startswith("%") else int(s) for s in srcs]
        regs[dst] = values[0] if op.startswith("mov") else _OPS[op.split(".")[0]](*values)
    return regs


def tile(extents: dict[str, int], raster=None, group=None, orient="m"):
    return SimpleNamespace(axes=tuple(axis(n, e) for n, e in extents.items()),
                           raster_axes=raster, raster_group=group, raster_orient=orient)


def decode(t, gids):
    b = PtxBuilder()
    gid = b.reg("b32")
    ints = _decode_axes(b, t, gid)
    lines = [line.strip() for line in b.body_lines()]
    out = []
    for g in gids:
        regs = run_int_ptx(lines, {gid: g})
        out.append(tuple(v if isinstance(v, int) else regs[v] for v in ints.values()))
    return list(ints), out


def test_without_a_group_the_decode_is_row_major():
    names, cells = decode(tile({"m": 2, "n": 3}), range(6))
    assert dict(zip(names, zip(*cells))) == {"m": (0, 0, 0, 1, 1, 1), "n": (0, 1, 2, 0, 1, 2)}


def test_a_group_of_two_walks_two_rows_before_moving_to_the_next_column():
    names, cells = decode(tile({"m": 4, "n": 3}, raster=("m", "n"), group=2), range(12))
    by_name = [dict(zip(names, c)) for c in cells]
    assert [(c["m"], c["n"]) for c in by_name] == [
        (0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2),
        (2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (3, 2)]


def test_a_ragged_last_group_is_narrower():
    names, cells = decode(tile({"m": 3, "n": 3}, raster=("m", "n"), group=2), range(9))
    by_name = [dict(zip(names, c)) for c in cells]
    assert [(c["m"], c["n"]) for c in by_name] == [
        (0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]


def test_inner_thread_axes_decode_as_usual_under_a_raster():
    t = tile({"m": 2, "n": 2, "u": 4}, raster=("m", "n"), group=2)
    names, cells = decode(t, range(16))
    by_name = [dict(zip(names, c)) for c in cells]
    assert [c["u"] for c in by_name] == [0, 1, 2, 3] * 4
    assert [(c["m"], c["n"]) for c in by_name[::4]] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert len({(c["m"], c["n"], c["u"]) for c in by_name}) == 16  # every cell exactly once


def test_orient_n_groups_the_second_axis():
    names, cells = decode(tile({"m": 2, "n": 4}, raster=("m", "n"), group=2, orient="n"), range(8))
    by_name = [dict(zip(names, c)) for c in cells]
    assert [(c["m"], c["n"]) for c in by_name] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
