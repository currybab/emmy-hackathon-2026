"""Merge neighbouring scalar shared-memory loads into ld.v2 / ld.v4 after the body is emitted.

An unrolled loop reads tile[row, k], tile[row, k+1], ... from one address register with
immediates 4 bytes apart; one vector load moves the same data in a quarter of the
instructions. Two conditions keep this exact:

- alignment: a v4 address must be 16-byte aligned and a v2 address 8-byte aligned, which the
  lowering proves per address register (base alignment and the index's known factor);
- no write in between: loads are merged only within a stretch with no store, atomic, barrier,
  cp.async, branch or label, so the merged load reads what each scalar load would have.

A load's destination register is always fresh, so defining it earlier is safe.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

# shared memory only: merging global loads made a01 and a02 slower when measured (60 -> 72 us
# and 33 -> 37 us), while shared tiles gained (m01 21 -> 15.7 us, now level with nvcc)
_SCALAR_LOAD = re.compile(r"^\tld\.(shared)\.f32 (%f\d+), \[(%rd\d+)(?:\+(-?\d+))?\];$")
# instructions after which memory may have changed or control may not fall through
_BOUNDARY = ("st.", "red.", "atom.", "bar.", "cp.async", "bra", "@", "ret", "membar", "fence")
F32_BYTES = 4
VECTOR_BYTES = {4: 16, 2: 8}  # lanes -> alignment the address needs


def _stretches(lines: Sequence[str]) -> list[tuple[int, int]]:
    """[start, end) index ranges free of any boundary."""
    spans, start = [], 0
    for i, line in enumerate(lines):
        text = line.strip()
        if text.endswith(":") or text.startswith(_BOUNDARY):
            spans.append((start, i))
            start = i + 1
    spans.append((start, len(lines)))
    return spans


def _merge_groups(loads: dict[int, tuple[int, str]], align: int) -> list[list[tuple[int, int, str]]]:
    """Runs of consecutive offsets that one aligned vector load can cover, largest first."""
    groups, taken = [], set()
    for offset in sorted(loads):
        if offset in taken:
            continue
        for lanes in (4, 2):
            need = VECTOR_BYTES[lanes]
            run = [offset + F32_BYTES * k for k in range(lanes)]
            if align >= need and offset % need == 0 and all(o in loads and o not in taken for o in run):
                groups.append([(o, *loads[o]) for o in run])
                taken.update(run)
                break
    return groups


def merge_loads(lines: Sequence[str], alignment: Mapping[str, int]) -> list[str]:
    out = list(lines)
    for start, end in _stretches(lines):
        by_base: dict[tuple[str, str], dict[int, tuple[int, str]]] = {}
        for i in range(start, end):
            match = _SCALAR_LOAD.match(lines[i])
            if match:
                space, reg, base, imm = match.groups()
                by_base.setdefault((space, base), {}).setdefault(int(imm or 0), (i, reg))
        for (space, base), loads in by_base.items():
            for group in _merge_groups(loads, alignment.get(base, 1)):
                first = min(index for _, index, _ in group)
                offset = group[0][0]
                regs = ", ".join(reg for _, _, reg in group)
                address = f"[{base}+{offset}]" if offset else f"[{base}]"
                for _, index, _ in group:
                    out[index] = None
                out[first] = f"\tld.{space}.v{len(group)}.f32 {{{regs}}}, {address};"
    return [line for line in out if line is not None]
