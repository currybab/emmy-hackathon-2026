"""Median kernel time of compile.py's PTX on the runner, optionally next to reference PTX.

    python scripts/bench.py                     # every open problem, ours only
    python scripts/bench.py --reference DIR     # also DIR/<problem>.ptx, e.g. nvcc -ptx output

Each kernel runs warmup 10 + 100 timed launches; the runner reports the median. submit.py
times a single cold launch, which is dominated by launch overhead for kernels of a few us.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from submit import RUNNER, post  # noqa: E402

WARMUP, ITERS = 10, 100


def median_us(ptx: str, kernel: dict) -> float:
    compiled = post(RUNNER, "/compile", {"ptx": ptx})
    if not compiled.get("ok"):
        raise SystemExit(compiled.get("stderr"))
    for _ in range(60):
        result = post(RUNNER, "/run", {
            "cubin_id": compiled["cubin_id"], "entry": kernel["entry"], "grid": kernel["grid"],
            "block": kernel["block"], "smem_bytes": kernel["smem_bytes"],
            "zero_outputs": kernel["zero_outputs"], "args": kernel["args"],
            "warmup": WARMUP, "iters": ITERS, "max_return": 1})
        if result.get("stage") != "busy":
            break
        time.sleep(2)
    if not result.get("ok"):
        raise SystemExit(result)
    return result["median_us"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, help="folder of <problem>.ptx to compare against")
    args = parser.parse_args()
    print(f"{'problem':22} {'ours us':>9}" + (f" {'ref us':>9} {'ratio':>7}" if args.reference else ""))
    for folder in sorted(p for p in (ROOT / "problems").iterdir() if p.is_dir()):
        kernel = json.loads((folder / "meta.json").read_text())["kernels"][0]
        ir = next(folder.glob("*.ir.json"))
        ours = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                              capture_output=True, text=True, check=True).stdout
        line = f"{folder.name:22} {median_us(ours, kernel):9.2f}"
        if args.reference:
            ours_us = float(line.split()[-1])
            ref_us = median_us((args.reference / f"{folder.name}.ptx").read_text(), kernel)
            line += f" {ref_us:9.2f} {ours_us / ref_us:6.2f}x"
        print(line)


if __name__ == "__main__":
    main()
