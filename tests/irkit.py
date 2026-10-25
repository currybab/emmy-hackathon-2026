"""Small constructors for hand-built Kernel IR in unit tests (real emmy classes, all fields named)."""
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write

from ptxc.builder import PtxBuilder
from ptxc.lower import BufferRef, Lowerer


def lit(v: int) -> Literal:
    return Literal(value=v, dtype="int")


def var(name: str) -> Var:
    return Var(name=name)


def binop(op: str, left, right) -> BinaryExpr:
    return BinaryExpr(op=op, left=left, right=right)


def axis(name: str, extent: int) -> Axis:
    return Axis(name=name, extent=Dim(lit(extent)), window=None)


def assign(name: str, op: str, *args: str) -> Assign:
    return Assign(name=name, op=ElementwiseImpl(op), args=tuple(args), dtype=F32)


def accum(name: str, value: str, op: str = "add", base: str | None = None, axes=("a0",)) -> Accum:
    return Accum(name=name, value=value, op=ElementwiseImpl(op), dtype=F32, axes=tuple(axes), base=base)


def load(name: str, buf: str, *index) -> Load:
    return Load(names=(name,), input=buf, index=tuple(index), dtype=F32)


def write(buf: str, value: str, *index) -> Write:
    return Write(output=buf, index=tuple(index), values=(value,), value_dtype=F32, atomic=False, swizzle="NONE")


def strided_loop(name: str, extent: int, start, step, body, seed: bool = True) -> StridedLoop:
    return StridedLoop(axis=axis(name, extent), start=start, step=step, body=tuple(body),
                       unroll=False, end=None, seed=seed)


def cond(predicate, body, else_body=()) -> Cond:
    return Cond(cond=predicate, body=tuple(body), else_body=tuple(else_body))


def smem(name: str, *extents: int) -> Smem:
    return Smem(name=name, extents=tuple(extents), dtype="float", align=0)


def sync() -> Sync:
    return Sync(barrier_id=0, count=None, warp=False)


def add_combine(state: str = "acc0") -> tuple:
    return (assign(state, "add", state, f"{state}__o"),)


def warp_shuffle(length: int = 32, state: str = "acc0") -> WarpShuffle:
    return WarpShuffle(state=(state,), state_b=(f"{state}__o",), combine_states=add_combine(state),
                       length=length, dtype=F32)


def tree_halve(buf: str, length: int, state: str = "acc0") -> TreeHalve:
    return TreeHalve(bufs=(buf,), state=(state,), state_b=(f"{state}__o",),
                     combine_states=add_combine(state), length=length, tid_var="warp", dtype=F32,
                     barrier_id=0, barrier_count=None, inner=None)


def lowerer(ints=None, buffers=("x", "out")) -> tuple[PtxBuilder, Lowerer]:
    """A Lowerer over 1-D global f32 buffers of 1024 elements each."""
    b = PtxBuilder()
    refs = {name: BufferRef(name, (1024,), b.reg("b64"), "global") for name in buffers}
    return b, Lowerer(b, refs, ints or {})


def body_text(b: PtxBuilder) -> list[str]:
    """Instructions and labels, comments dropped, whitespace stripped."""
    return [line.strip() for line in b.body_lines() if not line.strip().startswith("//")]
