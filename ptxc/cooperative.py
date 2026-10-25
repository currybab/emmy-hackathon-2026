"""Cooperative statements: the threads of a block combining their partial states.

    Sync         bar.sync - every thread of the block waits here
    WarpShuffle  XOR butterfly across the lanes of a warp, registers only
    TreeHalve    halving tree in shared memory across warps, result broadcast to all

Both combines run the IR's own `combine_states` (a list of Assigns) at every step instead of
assuming `+`, so the same code folds a sum, a max, or online softmax's (max, sum) pair.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ptxc.builder import PtxBuilder
from ptxc.errors import CompileError
from ptxc.expr import compare, int_binop

if TYPE_CHECKING:
    from ptxc.lower import Lowerer

WARP_SIZE = 32
FULL_MASK = "0xffffffff"  # every lane of the warp takes part
BFLY_CLAMP = "0x1f"  # shfl's c operand for a butterfly over the whole warp

_WARP_COMBINES = {"WarpShuffle", "TreeHalve"}


def uses_lane_warp(body: object) -> bool:
    """lane/warp exist iff the body holds a warp-level combine - the same rule emmy uses."""
    return any(
        type(stmt).__name__ in _WARP_COMBINES or any(uses_lane_warp(child) for child in stmt.nested())
        for stmt in body
    )


def lane_and_warp(b: PtxBuilder, tid: str) -> dict[str, str]:
    """lane = tid % 32 and warp = tid / 32, as a mask and a shift."""
    lane, warp = b.reg("b32"), b.reg("b32")
    b.comment("lane = tid & 31, warp = tid >> 5")
    b.emit(f"and.b32 {lane}, {tid}, {WARP_SIZE - 1}")
    b.emit(f"shr.u32 {warp}, {tid}, {WARP_SIZE.bit_length() - 1}")
    return {"lane": lane, "warp": warp}


def _halvings(length: int, what: str) -> list[int]:
    """length 32 -> [16, 8, 4, 2, 1]: the strides of a halving combine."""
    if length < 2 or length & (length - 1):
        raise CompileError(f"{what} over {length} is not a power of two of at least 2")
    strides = []
    stride = length // 2
    while stride:
        strides.append(stride)
        stride //= 2
    return strides


def barrier(b: PtxBuilder, barrier_id: int, count: int | None) -> None:
    b.emit(f"bar.sync {barrier_id}" if count is None else f"bar.sync {barrier_id}, {count}")


def lower_sync(lw: Lowerer, s: object) -> None:
    if s.warp:
        lw.b.emit(f"bar.warp.sync {FULL_MASK}")
    else:
        barrier(lw.b, s.barrier_id, s.count)


def lower_warp_shuffle(lw: Lowerer, s: object) -> None:
    """At each stride, every lane fetches its partner's state (lane ^ stride) and combines.

    After log2(length) steps every lane holds the combined value of its whole group.
    """
    if s.length > WARP_SIZE:
        raise CompileError(f"WarpShuffle over {s.length} lanes exceeds a warp of {WARP_SIZE}")
    for stride in _halvings(s.length, "WarpShuffle"):
        # fetch every state first: the combine reads all of them
        for state, partner in zip(s.state, s.state_b, strict=True):
            theirs = lw.b.reg("f32")
            lw.b.emit(f"shfl.sync.bfly.b32 {theirs}, {lw.value(state)}, {stride}, {BFLY_CLAMP}, {FULL_MASK}")
            lw.bind(partner, theirs)
        lw.lower_body(s.combine_states)


def lower_tree_halve(lw: Lowerer, s: object) -> None:
    """Fold `length` shared slots in halves, then give every thread slot 0.

        for stride in length/2 .. 1:
            if tid < stride:  slot[tid] = combine(slot[tid], slot[tid + stride])
            barrier
        state = slot[0]
    """
    if s.inner is not None:
        raise CompileError("TreeHalve with an inner level is not supported")
    tid = lw.int_value(s.tid_var)
    for stride in _halvings(s.length, "TreeHalve"):
        active = compare(lw.b, "<", tid, stride)
        if not isinstance(active, str):
            raise CompileError(f"TreeHalve index {s.tid_var!r} must vary across threads")
        skip = lw.b.label("tree_skip")
        lw.b.emit(f"@!{active} bra {skip}")
        partner = int_binop(lw.b, "+", tid, stride)
        for buf, state, other in zip(s.bufs, s.state, s.state_b, strict=True):
            lw.bind(state, lw.load_scalar(buf, tid))
            lw.bind(other, lw.load_scalar(buf, partner))
        lw.lower_body(s.combine_states)
        for buf, state in zip(s.bufs, s.state, strict=True):
            lw.store_scalar(buf, tid, lw.value(state))
        lw.b.place(skip)
        barrier(lw.b, s.barrier_id, s.barrier_count)
    for buf, state in zip(s.bufs, s.state, strict=True):
        lw.bind(state, lw.load_scalar(buf, 0))


# statement kind -> lowering function; the Lowerer registers these next to its own
HANDLERS = {
    "Sync": lower_sync,
    "WarpShuffle": lower_warp_shuffle,
    "TreeHalve": lower_tree_halve,
}
