"""PTX text builder: virtual registers, labels, and the final module layout.

Registers are virtual and never reused: every value gets a fresh one and ptxas does the real
register allocation. Counts are only known once the body is complete, so the `.reg`
declarations are written last, at the top of the entry, by `render`.
"""
from __future__ import annotations

import re
import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from ptxc.errors import CompileError

PTX_VERSION = "8.0"  # sm_89 needs 7.8 or later
PTX_TARGET = "sm_89"

# register class -> name prefix; the class doubles as the .reg declaration type
REG_PREFIX = {
    "pred": "%p",
    "b32": "%r",
    "b64": "%rd",
    "f32": "%f",
}

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+")


def f32_literal(value: float) -> str:
    """The exact PTX spelling of an f32 constant: `0f` followed by its IEEE-754 bits.

    Hex leaves no decimal-to-binary rounding to ptxas, so the kernel sees bit-for-bit the
    constant the IR meant.
    """
    (bits,) = struct.unpack(">I", struct.pack(">f", value))
    return f"0f{bits:08X}"


def check_identifier(name: str) -> str:
    if not _IDENTIFIER.fullmatch(name):
        raise CompileError(f"{name!r} is not a valid PTX identifier")
    return name


class PtxBuilder:
    """Accumulates one kernel's PTX text; the IR it is built from is never modified."""

    def __init__(self) -> None:
        self._reg_counts = {cls: 0 for cls in REG_PREFIX}
        self._label_count = 0
        self._lines: list[str] = []
        self._globals: list[str] = []
        # integer results already computed on every path to the current point, by what they compute
        self._known: dict[tuple, str] = {}
        self._non_negative: set[str] = set()

    # ---- reuse of integer work ------------------------------------------------------------

    def known(self, key: tuple) -> str | None:
        return self._known.get(key)

    def remember(self, key: tuple, reg: str) -> None:
        self._known[key] = reg

    @contextmanager
    def region(self) -> Iterator[None]:
        """Code that not every thread runs, or that runs again with new values (a branch, a
        loop body): what it computes cannot be reused once it is left."""
        outer = dict(self._known)
        try:
            yield
        finally:
            self._known = outer

    def mark_non_negative(self, reg: str) -> None:
        self._non_negative.add(reg)

    def is_non_negative(self, value: int | str) -> bool:
        return value >= 0 if isinstance(value, int) else value in self._non_negative

    def reg(self, cls: str) -> str:
        """A fresh virtual register of class `cls` (pred, b32, b64 or f32)."""
        if cls not in REG_PREFIX:
            raise CompileError(f"no PTX register class {cls!r}")
        self._reg_counts[cls] += 1
        return f"{REG_PREFIX[cls]}{self._reg_counts[cls]}"

    def label(self, hint: str) -> str:
        """A fresh label name; place it with `place`."""
        self._label_count += 1
        return f"$L_{hint}_{self._label_count}"

    def emit(self, instruction: str) -> None:
        """One instruction; the terminating `;` is added here."""
        self._lines.append(f"\t{instruction};")

    def place(self, label: str) -> None:
        self._lines.append(f"{label}:")

    def comment(self, text: str) -> None:
        # PTX rejects non-ASCII anywhere, comments included
        self._lines.append(f"\t// {text.encode('ascii', 'replace').decode()}")

    def declare_global(self, declaration: str) -> None:
        """A module-scope declaration, such as a `.shared` array."""
        self._globals.append(declaration)

    def body_lines(self) -> tuple[str, ...]:
        return tuple(self._lines)

    def render(self, entry: str, params: Sequence[str], max_threads: int) -> str:
        """The complete module: header, globals, signature, register declarations, body."""
        signature = ",\n".join(f"\t.param .u64 param_{check_identifier(p)}" for p in params)
        reg_decls = [
            f"\t.reg .{cls} {REG_PREFIX[cls]}<{count + 1}>;"
            for cls, count in self._reg_counts.items()
            if count
        ]
        return "\n".join([
            f".version {PTX_VERSION}",
            f".target {PTX_TARGET}",
            ".address_size 64",
            "",
            *self._globals,
            *([""] if self._globals else []),
            f".visible .entry {check_identifier(entry)}(",
            signature,
            ")",
            f".maxntid {max_threads}, 1, 1",
            "{",
            *reg_decls,
            "",
            *self._lines,
            "}",
            "",
        ])
