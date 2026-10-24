"""Statement lowering: one method per Kernel IR statement kind.

`Lowerer.lower_body` walks a statement sequence in order and dispatches on the statement's
class name. Two environments carry what earlier statements produced:

    ints    axis variables (a0, a1_co, ...) -> IntValue   (s32)
    values  SSA value names (in0, v1, ...)  -> f32 register
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ptxc.builder import PtxBuilder
from ptxc.errors import CompileError
from ptxc.expr import IntValue, int_binop, lower_int_expr
from ptxc.ops import emit_elementwise

F32_BYTES = 4
VECTOR_WIDTHS = (1, 2, 4)  # PTX has v2 and v4 memory ops; other group sizes go one by one


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


class Lowerer:
    def __init__(self, b: PtxBuilder, buffers: Mapping[str, BufferRef], ints: Mapping[str, IntValue]):
        self.b = b
        self.buffers = dict(buffers)
        self.ints: dict[str, IntValue] = dict(ints)
        self.values: dict[str, str] = {}
        self._handlers = {
            "Load": self._load,
            "Assign": self._assign,
            "Write": self._write,
        }

    def lower_body(self, body: Sequence[object]) -> None:
        for stmt in body:
            kind = type(stmt).__name__
            handler = self._handlers.get(kind)
            if handler is None:
                raise CompileError(f"statement {kind} is not supported")
            handler(stmt)

    # ---- lookups ---------------------------------------------------------------------------

    def _buffer(self, name: str) -> BufferRef:
        if name not in self.buffers:
            raise CompileError(f"buffer {name!r} is neither a kernel parameter nor shared memory")
        return self.buffers[name]

    def _value(self, name: str) -> str:
        if name not in self.values:
            raise CompileError(f"value {name!r} is used before it is defined")
        return self.values[name]

    # ---- addressing ------------------------------------------------------------------------

    def _flat_offset(self, buf: BufferRef, index: Sequence[object]) -> IntValue:
        """Row-major element offset of an index tuple: ((i0 * d1) + i1) * d2 + i2 ..."""
        if not index:
            return 0  # `sum_1[] = ...`: the buffer's only element
        if len(index) != len(buf.shape):
            raise CompileError(f"{buf.name}: {len(index)} indices for a {len(buf.shape)}-d buffer")
        offset: IntValue = 0
        for expr, dim in zip(index, buf.shape):
            scaled = int_binop(self.b, "*", offset, dim)
            offset = int_binop(self.b, "+", scaled, lower_int_expr(expr, self.b, self.ints))
        return offset

    def _addresses(self, buf: BufferRef, offset: IntValue, groups: list) -> list[str]:
        """The `[addr]` operand of each group; one 64-bit add serves them all."""
        if isinstance(offset, int):
            base, first = buf.base, offset
        else:
            byte_offset = self.b.reg("b64")
            base = self.b.reg("b64")
            self.b.emit(f"mul.wide.s32 {byte_offset}, {offset}, {F32_BYTES}")
            self.b.emit(f"add.s64 {base}, {buf.base}, {byte_offset}")
            first = 0
        operands = []
        for element, _ in groups:
            byte = (first + element) * F32_BYTES
            operands.append(f"[{base}+{byte}]" if byte else f"[{base}]")
        return operands

    # ---- statements ------------------------------------------------------------------------

    def _load(self, s: object) -> None:
        buf = self._buffer(s.input)
        self.b.comment(f"{', '.join(s.names)} = load {buf.name}")
        regs = [self.b.reg("f32") for _ in s.names]
        groups = _groups(regs)
        offset = self._flat_offset(buf, s.index)
        for (_, group), address in zip(groups, self._addresses(buf, offset, groups)):
            self.b.emit(f"ld.{buf.space}{_vector_suffix(len(group))}.f32 {_register_list(group)}, {address}")
        self.values.update(zip(s.names, regs))

    def _assign(self, s: object) -> None:
        dst = self.b.reg("f32")
        emit_elementwise(self.b, s.op.name, dst, [self._value(a) for a in s.args])
        self.values[s.name] = dst

    def _write(self, s: object) -> None:
        if s.atomic:
            raise CompileError(f"atomic write to {s.output!r} is not supported yet")
        buf = self._buffer(s.output)
        self.b.comment(f"{buf.name}[...] = {', '.join(s.values)}")
        regs = [self._value(v) for v in s.values]
        groups = _groups(regs)
        offset = self._flat_offset(buf, s.index)
        for (_, group), address in zip(groups, self._addresses(buf, offset, groups)):
            self.b.emit(f"st.{buf.space}{_vector_suffix(len(group))}.f32 {address}, {_register_list(group)}")
