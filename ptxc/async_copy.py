"""cp.async: copies from global to shared memory that run in the background.

    CpAsyncCopy    cp.async.{cg,ca}.shared.global [smem], [global], bytes
    CpAsyncCommit  cp.async.commit_group   - close the copies issued so far into one group
    CpAsyncWait    cp.async.wait_group n   - wait until at most n groups are still in flight

A tiled matmul issues the copies for a later K-slab, then computes on the current one while
they land; the wait plus a barrier makes a slab visible to the whole block before it is read.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ptxc.errors import CompileError
from ptxc.expr import affine_add, affine_scale, lower_affine

if TYPE_CHECKING:
    from ptxc.lower import Lowerer

CG_BYTES = 16  # .cg caches in L2 only and moves exactly 16 bytes; 4 and 8 go through L1 (.ca)
COPY_SIZES = (4, 8, 16)


def lower_copy(lw: Lowerer, s: object) -> None:
    if s.nbytes not in COPY_SIZES:
        raise CompileError(f"cp.async of {s.nbytes} bytes: PTX copies 4, 8 or 16")
    if s.swizzle != "NONE":
        raise CompileError(f"cp.async into {s.smem!r} with swizzle {s.swizzle} is not supported")
    smem, src = lw.buffer(s.smem), lw.buffer(s.src)
    dst_offset = lw.flat_offset(smem, s.smem_index)
    if s.lane_index is not None:
        # smem_index is the base of a stripe; the lane adds its own (row, col) inside it
        row, col = (lower_affine(e, lw.b, lw.ints) for e in s.lane_index)
        lane = affine_add(lw.b, affine_scale(lw.b, row, smem.shape[-1]), col)
        dst_offset = affine_add(lw.b, dst_offset, lane)
    dst = lw.address(smem, dst_offset)
    source = lw.address(src, lw.flat_offset(src, s.src_index))
    cache = "cg" if s.nbytes == CG_BYTES else "ca"
    lw.b.emit(f"cp.async.{cache}.shared.global {dst}, {source}, {s.nbytes}")


def lower_commit(lw: Lowerer, s: object) -> None:
    lw.b.emit("cp.async.commit_group")


def lower_wait(lw: Lowerer, s: object) -> None:
    lw.b.emit(f"cp.async.wait_group {s.group}")


# statement kind -> lowering function; the Lowerer registers these next to its own
HANDLERS = {
    "CpAsyncCopy": lower_copy,
    "CpAsyncCommit": lower_commit,
    "CpAsyncWait": lower_wait,
}
