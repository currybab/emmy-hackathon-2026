"""Kernel assembly: prologue (params, thread id, Tile decode, guard), body, epilogue."""
from __future__ import annotations

import math
from pathlib import Path

from ptxc.builder import PtxBuilder
from ptxc.cooperative import WARP_SIZE
from ptxc.errors import CompileError
from ptxc.expr import IntValue
from ptxc.loader import KernelSpec, load_kernel
from ptxc.lower import BufferRef, Lowerer


def compile_file(ir_path: Path) -> str:
    return compile_kernel(load_kernel(ir_path))


def compile_kernel(spec: KernelSpec) -> str:
    _check_launch(spec)
    tile = spec.tile
    b = PtxBuilder()

    buffers = _load_params(b, spec)
    tid, gid = _thread_ids(b)
    cells = _tile_cells(tile)
    ints = _decode_axes(b, tile, gid)
    if _uses_lane_warp(tile.body):
        ints = {**ints, **_lane_and_warp(b, tid)}

    exit_label = _guard(b, gid, cells) if spec.total_threads > cells else None

    Lowerer(b, buffers, ints).lower_body(tile.body)

    if exit_label:
        b.place(exit_label)
    b.emit("ret")
    return b.render(spec.name, [p.name for p in spec.params], max_threads=spec.block_threads)


def _check_launch(spec: KernelSpec) -> None:
    if spec.grid[1:] != (1, 1) or spec.block[1:] != (1, 1):
        raise CompileError(f"{spec.name}: only 1-D launches are supported, got "
                           f"grid {spec.grid} block {spec.block}")
    tile = spec.tile
    if tile.raster_axes is not None:
        raise CompileError(f"{spec.name}: rastered Tiles are not supported yet")
    if any(axis.window is not None for axis in tile.axes):
        raise CompileError(f"{spec.name}: windowed Tile axes are not supported yet")


def _load_params(b: PtxBuilder, spec: KernelSpec) -> dict[str, BufferRef]:
    """Each buffer parameter: load the pointer, convert it to a global-space address."""
    buffers = {}
    for param in spec.params:
        generic, base = b.reg("b64"), b.reg("b64")
        b.emit(f"ld.param.u64 {generic}, [param_{param.name}]")
        b.emit(f"cvta.to.global.u64 {base}, {generic}")
        buffers[param.name] = BufferRef(param.name, param.shape, base, "global")
    return buffers


def _thread_ids(b: PtxBuilder) -> tuple[str, str]:
    """(tid, gid): the thread's index in its block, and gid = blockIdx.x * blockDim.x + tid."""
    ctaid, ntid, tid, gid = (b.reg("b32") for _ in range(4))
    b.comment("gid = ctaid.x * ntid.x + tid.x")
    b.emit(f"mov.u32 {ctaid}, %ctaid.x")
    b.emit(f"mov.u32 {ntid}, %ntid.x")
    b.emit(f"mov.u32 {tid}, %tid.x")
    b.emit(f"mad.lo.s32 {gid}, {ctaid}, {ntid}, {tid}")
    return tid, gid


_WARP_COMBINES = {"WarpShuffle", "TreeHalve"}


def _uses_lane_warp(body: object) -> bool:
    """lane/warp exist iff the body holds a warp-level combine - the same rule emmy uses."""
    return any(
        type(stmt).__name__ in _WARP_COMBINES or any(_uses_lane_warp(child) for child in stmt.nested())
        for stmt in body
    )


def _lane_and_warp(b: PtxBuilder, tid: str) -> dict[str, str]:
    """lane = tid % 32 and warp = tid / 32, as a mask and a shift."""
    lane, warp = b.reg("b32"), b.reg("b32")
    b.comment("lane = tid & 31, warp = tid >> 5")
    b.emit(f"and.b32 {lane}, {tid}, {WARP_SIZE - 1}")
    b.emit(f"shr.u32 {warp}, {tid}, {WARP_SIZE.bit_length() - 1}")
    return {"lane": lane, "warp": warp}


def _tile_cells(tile: object) -> int:
    return math.prod(axis.extent.as_static() for axis in tile.axes)


def _decode_axes(b: PtxBuilder, tile: object, gid: str) -> dict[str, IntValue]:
    """Split gid into the Tile's axis indices, row-major: the last axis varies fastest.

    Tile[a0, a1_co] with extents (64, 128):  a1_co = gid % 128,  a0 = gid / 128
    """
    ints: dict[str, IntValue] = {}
    rest = gid
    for axis in reversed(tile.axes[1:]):
        extent = axis.extent.as_static()
        index, quotient = b.reg("b32"), b.reg("b32")
        b.emit(f"rem.u32 {index}, {rest}, {extent}")
        b.emit(f"div.u32 {quotient}, {rest}, {extent}")
        ints[axis.name] = index
        rest = quotient
    ints[tile.axes[0].name] = rest
    b.comment("Tile axes: " + ", ".join(f"{name} = {reg}" for name, reg in ints.items()))
    return ints


def _guard(b: PtxBuilder, gid: str, cells: int) -> str:
    """The launch has more threads than the Tile has cells: the extra ones exit at once."""
    # safe to exit early: only pointwise Tiles are over-launched, and they never synchronize
    out_of_range, exit_label = b.reg("pred"), b.label("exit")
    b.comment(f"guard: only the first {cells} threads have a cell")
    b.emit(f"setp.ge.u32 {out_of_range}, {gid}, {cells}")
    b.emit(f"@{out_of_range} bra {exit_label}")
    return exit_label
