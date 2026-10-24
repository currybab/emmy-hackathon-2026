#!/usr/bin/env python3
"""Emmy Kernel IR -> PTX.

    python compile.py path/to/kernel.ir.json > out.ptx

Reads the IR and the meta.json beside it, writes PTX to stdout. On failure, writes the reason
to stderr and exits non-zero - never a partial or guessed PTX.
"""
import sys
from pathlib import Path

from ptxc.compiler import compile_file
from ptxc.errors import CompileError

USAGE = "usage: python compile.py path/to/kernel.ir.json > out.ptx"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        ptx = compile_file(Path(argv[1]))
    except CompileError as exc:
        print(f"compile.py: error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
