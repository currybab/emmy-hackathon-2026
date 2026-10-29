"""Kernel assembly: prologue (params, thread id, Tile decode, guard), body, epilogue."""
from __future__ import annotations

import math
from pathlib import Path

from ptxc import cooperative
from ptxc.builder import PtxBuilder
from ptxc.errors import CompileError
from ptxc.expr import IntValue, int_binop, materialize
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
    if cooperative.uses_lane_warp(tile.body):
        ints = {**ints, **cooperative.lane_and_warp(b, tid)}

    exit_label = _guard(b, gid, cells) if spec.total_threads > cells else None

    Lowerer(b, buffers, ints, smem_pool_bytes=spec.smem_bytes).lower_body(tile.body)

    if exit_label:
        b.place(exit_label)
    b.emit("ret")
    return b.render(spec.name, [p.name for p in spec.params], max_threads=spec.block_threads)


def _check_launch(spec: KernelSpec) -> None:
    if spec.grid[1:] != (1, 1) or spec.block[1:] != (1, 1):
        raise CompileError(f"{spec.name}: only 1-D launches are supported, got "
                           f"grid {spec.grid} block {spec.block}")
    tile = spec.tile
    if tile.aux_threads:
        raise CompileError(f"{spec.name}: warp-specialized Tiles (aux threads) are not supported")


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
    for reg in (tid, gid):
        b.mark_non_negative(reg)
    return tid, gid


def _tile_cells(tile: object) -> int:
    return math.prod(axis.extent.as_static() for axis in tile.axes)


def _decode_axes(b: PtxBuilder, tile: object, gid: str) -> dict[str, IntValue]:
    """Split gid into the Tile's axis indices, row-major: the last axis varies fastest.

    Tile[a0, a1_co] with extents (64, 128):  a1_co = gid % 128,  a0 = gid / 128
    """
    raster = _raster(tile)
    if raster is not None:
        return _decode_rastered(b, tile, gid, *raster)
    ints: dict[str, IntValue] = {}
    rest = gid
    for axis in reversed(tile.axes[1:]):
        extent = axis.extent.as_static()
        index, quotient = b.reg("b32"), b.reg("b32")
        b.emit(f"rem.u32 {index}, {rest}, {extent}")
        b.emit(f"div.u32 {quotient}, {rest}, {extent}")
        b.mark_non_negative(index)
        b.mark_non_negative(quotient)
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


def _raster(tile: object) -> tuple[int, int, int] | None:
    """(grouped axis, other axis, group size) when the Tile reorders its blocks, else None.

    `raster_axes` names the block axes (m, n), adjacent in the Tile; `raster_orient` says which
    one is grouped. A group of one block, or an axis of one block, changes nothing.
    """
    if not tile.raster_group or not tile.raster_axes:
        return None
    names = [axis.name for axis in tile.axes]
    m, n = tile.raster_axes
    if m not in names or n not in names or names.index(n) != names.index(m) + 1:
        return None
    mi, ni = names.index(m), names.index(n)
    grouped, other = (mi, ni) if tile.raster_orient == "m" else (ni, mi)
    extent = tile.axes[grouped].extent.as_static()
    group = min(tile.raster_group, extent)
    return (grouped, other, group) if extent > 1 and group > 1 else None


def _decode_rastered(b: PtxBuilder, tile: object, gid: str, grouped: int, other: int,
                     group: int) -> dict[str, IntValue]:
    """Row-major decode, except that the two block axes walk `group` blocks of the grouped
    axis before stepping the other one: neighbouring blocks then read the same slab of the
    shared operand while it is still in L2 (the standard threadblock swizzle).

        sub = flat (grouped, other) block id;  span = group * other_extent
        grouped = (sub / span) * group + (sub % span) % width
        other   = (sub % span) / width         width = min(group, extent - first row of the group)
    """
    extents = [axis.extent.as_static() for axis in tile.axes]
    strides = [math.prod(extents[i + 1:]) for i in range(len(extents))]
    first, second = min(grouped, other), max(grouped, other)
    b.comment(f"block raster: groups of {group} along {tile.axes[grouped].name}")
    sub = int_binop(b, "/", gid, strides[second])
    if first != 0:
        sub = int_binop(b, "%", sub, extents[grouped] * extents[other])
    span = group * extents[other]
    group_index = int_binop(b, "/", sub, span)
    within = int_binop(b, "%", sub, span)
    if extents[grouped] % group == 0:
        width: IntValue = group
    else:  # the last group is ragged
        left = int_binop(b, "-", extents[grouped], int_binop(b, "*", group_index, group))
        width = b.reg("b32")
        b.emit(f"min.s32 {width}, {materialize(b, left)}, {group}")
    ints: dict[str, IntValue] = {}
    for i, axis in enumerate(tile.axes):
        if i == grouped:
            start = int_binop(b, "*", group_index, group)
            ints[axis.name] = int_binop(b, "+", start, int_binop(b, "%", within, width))
        elif i == other:
            ints[axis.name] = int_binop(b, "/", within, width)
        else:
            index = int_binop(b, "/", gid, strides[i])
            ints[axis.name] = index if i == 0 else int_binop(b, "%", index, extents[i])
    b.comment("Tile axes: " + ", ".join(f"{name} = {value}" for name, value in ints.items()))
    return ints
