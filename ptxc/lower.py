"""Statement lowering: one method per Kernel IR statement kind.

`Lowerer.lower_body` walks a statement sequence in order and dispatches on the statement's
class name. Two environments carry what earlier statements produced:

    ints    axis variables (a0, a1_co, lane, ...) -> IntValue   (s32)
    values  SSA value names (in0, acc0, ...)       -> f32 register

This module covers data movement and control flow. Statements in which threads cooperate
(Sync, WarpShuffle, TreeHalve) are lowered by cooperative.py, and cp.async copies by
async_copy.py, both through the public methods here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

from ptxc import async_copy, cooperative
from ptxc.builder import PtxBuilder, check_identifier, f32_literal
from ptxc.errors import CompileError
from ptxc.expr import Affine, IntValue, affine_add, affine_scale, as_affine, lower_affine, lower_int_expr, lower_predicate
from ptxc.elementwise import emit_elementwise

F32_BYTES = 4
VECTOR_WIDTHS = (1, 2, 4)  # PTX has v2 and v4 memory ops; other group sizes go one by one
MAX_UNROLL = 128  # an unrolled Loop longer than this stays a loop
SMEM_POOL = "smem_pool"
POOL_ALIGN = 16


@dataclass(frozen=True)
class BufferRef:
    """A buffer as the kernel sees it: its shape and the register holding its base address."""
    name: str
    shape: tuple[int, ...]
    base: str
    space: str  # "global" or "shared"


def _vector_suffix(width: int) -> str:
    return "" if width == 1 else f".v{width}"


def _register_list(regs: Sequence[str]) -> str:
    return regs[0] if len(regs) == 1 else "{" + ", ".join(regs) + "}"


def _groups(regs: Sequence[str]) -> list[tuple[int, Sequence[str]]]:
    """Split consecutive values into (element offset, registers) groups PTX can move at once."""
    if len(regs) in VECTOR_WIDTHS:
        return [(0, regs)]
    return [(i, [reg]) for i, reg in enumerate(regs)]


def _describe(stmt: object) -> str:
    """The statement's first line in emmy's own IR notation, for a PTX comment."""
    pretty = getattr(stmt, "pretty", None)
    return pretty()[0].strip() if pretty else type(stmt).__name__


class Lowerer:
    def __init__(self, b: PtxBuilder, buffers: Mapping[str, BufferRef], ints: Mapping[str, IntValue],
                 smem_pool_bytes: int = 0):
        self.b = b
        self.buffers = dict(buffers)
        self.ints: dict[str, IntValue] = dict(ints)
        self.values: dict[str, str] = {}
        # > 0: the launch gives this much dynamic shared memory and Smem arrays live in it
        self.smem_pool_bytes = smem_pool_bytes
        self._pool_used = 0
        self._handlers = {
            "Load": self._load,
            "Assign": self._assign,
            "Let": self._let,
            "Init": self._init,
            "Write": self._write,
            "Accum": self._accum,
            "StridedLoop": self._strided_loop,
            "Loop": self._loop,
            "Cond": self._cond,
            "Smem": self._smem,
            **{kind: partial(fn, self) for kind, fn in cooperative.HANDLERS.items()},
            **{kind: partial(fn, self) for kind, fn in async_copy.HANDLERS.items()},
        }

    def lower_body(self, body: Sequence[object]) -> None:
        for stmt in body:
            kind = type(stmt).__name__
            handler = self._handlers.get(kind)
            if handler is None:
                raise CompileError(f"statement {kind} is not supported")
            self.b.comment(_describe(stmt))
            handler(stmt)

    # ---- environment -----------------------------------------------------------------------

    def value(self, name: str) -> str:
        if name not in self.values:
            raise CompileError(f"value {name!r} is used before it is defined")
        return self.values[name]

    def bind(self, name: str, reg: str) -> None:
        self.values[name] = reg

    def int_value(self, name: str) -> IntValue:
        if name not in self.ints:
            raise CompileError(f"index variable {name!r} is not defined here")
        return self.ints[name]

    def buffer(self, name: str) -> BufferRef:
        if name not in self.buffers:
            raise CompileError(f"buffer {name!r} is neither a kernel parameter nor shared memory")
        return self.buffers[name]

    # ---- memory ----------------------------------------------------------------------------

    def load_scalar(self, buf_name: str, offset: IntValue) -> str:
        """A fresh f32 register loaded from element `offset` of a buffer."""
        reg = self.b.reg("f32")
        self._move("ld", self.buffer(buf_name), as_affine(offset), [reg])
        return reg

    def store_scalar(self, buf_name: str, offset: IntValue, reg: str) -> None:
        self._move("st", self.buffer(buf_name), as_affine(offset), [reg])

    def _move(self, kind: str, buf: BufferRef, offset: Affine, regs: Sequence[str]) -> None:
        """ld or st of consecutive f32 values starting at element `offset`."""
        groups = _groups(regs)
        for (_, group), address in zip(groups, self._addresses(buf, offset, groups)):
            op = f"{kind}.{buf.space}{_vector_suffix(len(group))}.f32"
            values = _register_list(group)
            self.b.emit(f"{op} {values}, {address}" if kind == "ld" else f"{op} {address}, {values}")

    def flat_offset(self, buf: BufferRef, index: Sequence[object]) -> Affine:
        """Row-major element offset of an index tuple: ((i0 * d1) + i1) * d2 + i2 ...

        Kept as (variable, constant) so the constant becomes the address immediate.
        """
        if not index:
            return (0, 0)  # `sum_1[] = ...`: the buffer's only element
        if len(index) != len(buf.shape):
            raise CompileError(f"{buf.name}: {len(index)} indices for a {len(buf.shape)}-d buffer")
        offset: Affine = (0, 0)
        for expr, dim in zip(index, buf.shape):
            offset = affine_add(self.b, affine_scale(self.b, offset, dim), lower_affine(expr, self.b, self.ints))
        return offset

    def address(self, buf: BufferRef, offset: Affine) -> str:
        """The `[addr]` operand of element `offset` of a buffer."""
        return self._addresses(buf, offset, [(0, None)])[0]

    def _element_address(self, buf: BufferRef, index: str) -> str:
        """base + 4 * index as a 64-bit address, computed once per (buffer, index)."""
        key = ("address", buf.base, index)
        known = self.b.known(key)
        if known is not None:
            return known
        byte_offset = self.b.reg("b64")
        address = self.b.reg("b64")
        self.b.emit(f"mul.wide.s32 {byte_offset}, {index}, {F32_BYTES}")
        self.b.emit(f"add.s64 {address}, {buf.base}, {byte_offset}")
        self.b.remember(key, address)
        return address

    def _addresses(self, buf: BufferRef, offset: Affine, groups: list) -> list[str]:
        """The `[addr]` operand of each group; one 64-bit add serves them all."""
        variable, first = offset
        if isinstance(variable, int):
            base, first = buf.base, variable + first
        else:
            base = self._element_address(buf, variable)
        operands = []
        for element, _ in groups:
            byte = (first + element) * F32_BYTES
            operands.append(f"[{base}+{byte}]" if byte else f"[{base}]")
        return operands

    # ---- data statements -------------------------------------------------------------------

    def _load(self, s: object) -> None:
        regs = [self.b.reg("f32") for _ in s.names]
        self._move("ld", self.buffer(s.input), self.flat_offset(self.buffer(s.input), s.index), regs)
        for name, reg in zip(s.names, regs):
            self.bind(name, reg)

    def _assign(self, s: object) -> None:
        dst = self.b.reg("f32")
        emit_elementwise(self.b, s.op.name, dst, [self.value(a) for a in s.args])
        self.bind(s.name, dst)

    def _let(self, s: object) -> None:
        """A named f32 constant (`f32 one = 1.0`): one mov of its bit pattern."""
        literal = s.value
        if type(literal).__name__ != "Literal" or not isinstance(literal.value, float):
            raise CompileError(f"Let {s.name!r}: only float literals are supported, got {literal!r}")
        reg = self.b.reg("f32")
        self.b.emit(f"mov.f32 {reg}, {f32_literal(literal.value)}")
        self.bind(s.name, reg)

    def _init(self, s: object) -> None:
        """An accumulator set up before the code that folds into it (`Init(f32 acc = 0.0)`)."""
        reg = self.b.reg("f32")
        self.b.emit(f"mov.f32 {reg}, {f32_literal(s.identity)}")
        self.bind(s.name, reg)

    def _write(self, s: object) -> None:
        buf = self.buffer(s.output)
        offset = self.flat_offset(buf, s.index)
        regs = [self.value(v) for v in s.values]
        if not s.atomic:
            self._move("st", buf, offset, regs)
            return
        # `out[i] += v (atomic)`: several blocks add partial sums into the same element.
        # red is atom without a result - nobody reads the old value back.
        groups = [(i, [reg]) for i, reg in enumerate(regs)]
        for (_, (reg,)), address in zip(groups, self._addresses(buf, offset, groups)):
            self.b.emit(f"red.{buf.space}.add.f32 {address}, {reg}")

    def _accum(self, s: object) -> None:
        """acc = op(acc, value), or op(base, value) when the IR names a base; updated in place."""
        if s.name not in self.values:
            raise CompileError(f"accumulator {s.name!r} is folded before it is seeded")
        acc = self.values[s.name]
        first = self.value(s.base) if s.base is not None else acc
        emit_elementwise(self.b, s.op.name, acc, [first, self.value(s.value)])

    def _smem(self, s: object) -> None:
        """A static __shared__ array: declared at module scope, addressed through a register."""
        if s.dtype != "float":
            raise CompileError(f"shared array {s.name!r} has unsupported dtype {s.dtype!r}")
        if s.name in self.buffers:
            raise CompileError(f"shared array {s.name!r} is declared twice")
        extents = tuple(int(e) for e in s.extents)
        nbytes = math.prod(extents) * F32_BYTES
        align = max(s.align, F32_BYTES)
        base = self.b.reg("b64")
        if self.smem_pool_bytes:
            self._carve_from_pool(s.name, base, nbytes, align)
        else:
            self.b.declare_global(f".shared .align {align} .b8 {check_identifier(s.name)}[{nbytes}];")
            self.b.emit(f"mov.u64 {base}, {s.name}")
        self.buffers[s.name] = BufferRef(s.name, extents, base, "shared")

    def _carve_from_pool(self, name: str, base: str, nbytes: int, align: int) -> None:
        """Past 48 KB a kernel cannot declare shared arrays statically: the launch hands it a
        dynamic pool (meta.json `smem_bytes`) and each array takes the next aligned slice."""
        if align > POOL_ALIGN:
            raise CompileError(f"shared array {name!r} wants {align}-byte alignment; the pool has {POOL_ALIGN}")
        offset = -(-self._pool_used // align) * align  # round up to the alignment
        if offset + nbytes > self.smem_pool_bytes:
            raise CompileError(f"shared array {name!r} does not fit the {self.smem_pool_bytes}-byte "
                               f"dynamic pool ({offset + nbytes} bytes needed)")
        if self._pool_used == 0:
            self.b.declare_global(f".extern .shared .align {POOL_ALIGN} .b8 {SMEM_POOL}[];")
        self._pool_used = offset + nbytes
        self.b.emit(f"mov.u64 {base}, {SMEM_POOL}")
        if offset:
            self.b.emit(f"add.s64 {base}, {base}, {offset}")

    # ---- control flow ----------------------------------------------------------------------

    def _seed_accumulators(self, body: Sequence[object], seed: bool) -> None:
        """Before a loop: each Accum directly in its body starts at the op's identity.

        Only the immediate body counts - a nested loop seeds its own (emmy's rule). With
        `seed=False` the accumulator was set up outside the loop and is left alone.
        """
        seeded = set()
        for stmt in body:
            if type(stmt).__name__ != "Accum" or stmt.name in seeded:
                continue
            seeded.add(stmt.name)
            if not seed:
                continue
            if stmt.op.identity is None:
                raise CompileError(f"accumulator {stmt.name!r}: {stmt.op.name!r} has no identity")
            reg = self.b.reg("f32")
            self.b.emit(f"mov.f32 {reg}, {f32_literal(stmt.op.identity)}")
            self.bind(stmt.name, reg)

    def _strided_loop(self, s: object) -> None:
        """for (axis = start; axis < end; axis += step) body   - end defaults to the axis extent."""
        self._seed_accumulators(s.body, s.seed)
        start = lower_int_expr(s.start, self.b, self.ints)
        step = s.step if isinstance(s.step, int) else lower_int_expr(s.step, self.b, self.ints)
        end = s.axis.extent.as_static() if s.end is None else lower_int_expr(s.end, self.b, self.ints)
        counter = self.b.reg("b32")
        self.b.emit(f"mov.s32 {counter}, {start}")
        if self.b.is_non_negative(start) and isinstance(step, int) and step > 0:
            self.b.mark_non_negative(counter)  # it only counts up from there
        self._counted_loop(s.axis.name, counter, end, step, s.body)

    def _loop(self, s: object) -> None:
        """for (axis = 0; axis < extent; axis++) body - unrolled when the IR asks for it."""
        self._seed_accumulators(s.body, s.seed)
        extent = s.axis.extent.as_static()
        if s.unroll and extent <= MAX_UNROLL:
            self._unrolled(s.axis.name, extent, s.body)
            return
        counter = self.b.reg("b32")
        self.b.emit(f"mov.s32 {counter}, 0")
        self.b.mark_non_negative(counter)
        self._counted_loop(s.axis.name, counter, extent, 1, s.body)

    def _unrolled(self, var: str, extent: int, body: Sequence[object]) -> None:
        """One copy of the body per iteration, with the axis variable a compile-time int:
        index math on it folds away, so `tile[k][col]` becomes base + col + an immediate."""
        outer = self.ints
        for i in range(extent):
            self.b.comment(f"{var} = {i}")
            self.ints = {**outer, var: i}
            self.lower_body(body)
        self.ints = outer

    def _counted_loop(self, var: str, counter: str, end: IntValue, step: IntValue,
                      body: Sequence[object]) -> None:
        """Top-tested loop; inside the body the axis variable `var` is the counter register."""
        top, done = self.b.label(f"{var}_loop"), self.b.label(f"{var}_done")
        self.b.place(top)
        finished = self.b.reg("pred")
        self.b.emit(f"setp.ge.s32 {finished}, {counter}, {end}")
        self.b.emit(f"@{finished} bra {done}")
        outer = self.ints
        self.ints = {**outer, var: counter}
        with self.b.region():
            self.lower_body(body)
        self.ints = outer
        self.b.emit(f"add.s32 {counter}, {counter}, {step}")
        self.b.emit(f"bra {top}")
        self.b.place(done)

    def _cond(self, s: object) -> None:
        """if (cond) body else else_body; a predicate known at compile time keeps one branch."""
        holds = lower_predicate(s.cond, self.b, self.ints)
        if holds is True or holds is False:
            self.lower_body(s.body if holds else s.else_body)
            return
        end = self.b.label("endif")
        skip = self.b.label("else") if s.else_body else end
        self.b.emit(f"@!{holds} bra {skip}")
        with self.b.region():
            self.lower_body(s.body)
        if s.else_body:
            self.b.emit(f"bra {end}")
            self.b.place(skip)
            with self.b.region():
                self.lower_body(s.else_body)
        self.b.place(end)
