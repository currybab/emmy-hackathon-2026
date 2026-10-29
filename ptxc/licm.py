"""Loop-invariant code motion over a loop body's PTX lines.

attention's q row is read again in every iteration of the loop over keys although it never
changes; nvcc's front end moves such work before the loop, and ptxas does not. An instruction
moves when it computes the same value in every iteration:

- it is pure (arithmetic, conversion, select) or, where allowed, a global load in a loop
  that writes no global memory;
- every register it reads is defined outside the loop or by an instruction already moved;
- its destination is defined once in the loop and not read before that definition (which
  would be the previous iteration's value);
- it is not inside a branch or an inner loop of the body, where it might not run.

Loads move only when the caller allows them: a loop that may run zero times must not gain a
load it would never have made.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

_REGISTER = re.compile(r"%(?:rd|r|f|p)\d+")
_LABEL_REF = re.compile(r"(\$L\w+)")
_PURE = ("add.", "sub.", "mul.", "mad.", "fma.", "div.", "rem.", "shl.", "shr.", "and.", "or.", "xor.",
         "neg.", "abs.", "min.", "max.", "cvt.", "mov.", "selp.", "setp.", "rcp.", "sqrt.", "rsqrt.",
         "ex2.", "lg2.", "copysign.")
_GLOBAL_LOAD = "ld.global."
_GLOBAL_WRITE = ("st.global", "red.global", "atom.global")


def _defs_and_uses(text: str) -> tuple[list[str], list[str]]:
    """Registers an instruction writes and reads. Loads write the registers before the
    address; everything else writes its first operand."""
    if text.startswith("ld."):
        before, _, address = text.partition("[")
        return _REGISTER.findall(before), _REGISTER.findall(address)
    if text.startswith(("st.", "red.", "atom.")):
        return [], _REGISTER.findall(text)
    regs = _REGISTER.findall(text)
    return regs[:1], regs[1:]


def _instructions(lines: Sequence[str]) -> Iterable[tuple[int, str]]:
    for i, line in enumerate(lines):
        text = line.strip()
        if text and not text.startswith("//") and not text.endswith(":"):
            yield i, text.rstrip(";")


def hoist_invariants(lines: Sequence[str], variant: set[str],
                     allow_loads: bool) -> tuple[list[str], list[str]]:
    """Split a loop body into (lines to place before the loop, lines that stay), both in order.
    `variant` names registers the loop itself changes, such as its counter."""
    defs = Counter(reg for _, text in _instructions(lines) for reg in _defs_and_uses(text)[0])
    defined_inside = set(defs) | variant
    loads_ok = allow_loads and not any(text.startswith(_GLOBAL_WRITE) for _, text in _instructions(lines))

    moved: set[int] = set()
    invariant: set[str] = set()
    read_so_far: set[str] = set()
    pending_targets: set[str] = set()  # labels of open branches: code before them may be skipped
    for i, line in enumerate(lines):
        text = line.strip()
        if text.endswith(":"):
            pending_targets.discard(text[:-1])
            continue
        if not text or text.startswith("//"):
            continue
        text = text.rstrip(";")
        if "bra" in text.split()[0] or text.startswith("@"):
            later = {m for m in _LABEL_REF.findall(text) if any(l.strip() == f"{m}:" for l in lines[i + 1:])}
            pending_targets |= later
        dsts, srcs = _defs_and_uses(text)
        movable = (
            not pending_targets
            and not text.startswith("@")
            and (text.startswith(_PURE) or (loads_ok and text.startswith(_GLOBAL_LOAD)))
            and dsts and all(defs[d] == 1 and d not in variant and d not in read_so_far for d in dsts)
            and all(s not in defined_inside or s in invariant for s in srcs)
        )
        if movable:
            moved.add(i)
            invariant.update(dsts)
        read_so_far.update(srcs)
    hoisted = [line for i, line in enumerate(lines) if i in moved]
    rest = [line for i, line in enumerate(lines) if i not in moved]
    return hoisted, rest
