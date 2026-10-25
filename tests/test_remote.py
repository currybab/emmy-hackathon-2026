"""End to end: compile.py -> hosted runner (ptxas + RTX 5090) -> compare with expected.npz.

Opt-in because it needs the network:  PTXC_REMOTE=1 pytest -m remote
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = [
    "p00_scalar_add", "p01_vector_add", "p02_scale_shift", "p03_relu_mul",  # 1 pointwise
    "r01_array_sum", "r02_big_sum", "r03_mean", "r04_row_sum",  # 2 reductions
]

pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(os.environ.get("PTXC_REMOTE") != "1", reason="set PTXC_REMOTE=1 to use the runner"),
]


@pytest.mark.parametrize("problem", PROBLEMS)
def test_problem_passes_on_the_runner(problem):
    r = subprocess.run(
        [sys.executable, "submit.py", problem, "--compiler", "compile.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr
