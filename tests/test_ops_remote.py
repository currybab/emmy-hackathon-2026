"""Every elementwise op on the GPU against numpy in float64 (opt-in: PTXC_REMOTE=1).

Each case is a one-op pointwise kernel built as Kernel IR text, compiled by our compiler and
run on the runner over 1024 standard-normal inputs x0, x1 (seeds 0 and 1, as the runner makes
them). Run with -s to see each op's error.
"""
import json
import os

import numpy as np
import pytest

from ptxc.compiler import compile_file
from submit import RUNNER, post

pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(os.environ.get("PTXC_REMOTE") != "1", reason="set PTXC_REMOTE=1 to use the runner"),
]

N = 1024
F32 = "DataType(name='f32', np=dtype('float32'), nbytes=4)"
x0 = np.random.default_rng(0).standard_normal(N).astype(np.float32).astype(np.float64)
x1 = np.random.default_rng(1).standard_normal(N).astype(np.float32).astype(np.float64)


def erf(x):
    from math import erf as scalar_erf
    return np.vectorize(scalar_erf)(x)


# name -> (statements as (result, op, args), the expected value from x0/x1 in float64)
# `k` is a Let constant when a case needs one: ("k", "let", value)
CASES = {
    "exp": ([("v", "exp", ("in0",))], np.exp(x0)),
    # inputs up to about +-100: e^x reaches 0 and overflows. The runner's JSON sends inf (and
    # NaN) as null, so overflow is checked on the GPU: e > 3e38 holds only for inf
    "exp_wide": ([("k", "let", 30.0), ("s", "multiply", ("in0", "k")), ("e", "exp", ("s",)),
                  ("m", "let", 3e38), ("v", "minimum", ("e", "m"))],
                 np.minimum(np.exp(np.float32(x0 * 30.0).astype(np.float64)), 3e38)),
    "exp_overflow_is_inf": ([("k", "let", 30.0), ("s", "multiply", ("in0", "k")), ("e", "exp", ("s",)),
                             ("m", "let", 3e38), ("v", "greater", ("e", "m"))],
                            (np.float32(x0 * 30.0) > 88.73).astype(np.float64)),
    "exp_fast": ([("v", "exp_fast", ("in0",))], np.exp(x0)),
    "log": ([("a", "abs", ("in0",)), ("v", "log", ("a",))], np.log(np.abs(x0))),
    "sigmoid": ([("v", "sigmoid", ("in0",))], 1 / (1 + np.exp(-x0))),
    "tanh": ([("v", "tanh", ("in0",))], np.tanh(x0)),
    "tanh_small": ([("k", "let", 0.001), ("s", "multiply", ("in0", "k")), ("v", "tanh", ("s",))],
                   np.tanh(np.float32(x0 * 0.001).astype(np.float64))),
    "erf": ([("v", "erf", ("in0",))], erf(x0)),
    "sin": ([("k", "let", 10.0), ("s", "multiply", ("in0", "k")), ("v", "sin", ("s",))],
            np.sin(np.float32(x0 * 10.0).astype(np.float64))),
    "cos": ([("k", "let", 10.0), ("s", "multiply", ("in0", "k")), ("v", "cos", ("s",))],
            np.cos(np.float32(x0 * 10.0).astype(np.float64))),
    "pow": ([("a", "abs", ("in0",)), ("v", "pow", ("a", "in1"))], np.abs(x0) ** x1),
    "pow_negative_base": ([("k", "let", 3.0), ("v", "pow", ("in0", "k"))], x0 ** 3),
    "rsqrt": ([("a", "abs", ("in0",)), ("v", "rsqrt", ("a",))], 1 / np.sqrt(np.abs(x0))),
    "minimum": ([("v", "minimum", ("in0", "in1"))], np.minimum(x0, x1)),
    "negative": ([("v", "negative", ("in0",))], -x0),
    "square": ([("v", "square", ("in0",))], np.float32(x0 * x0).astype(np.float64)),
    "greater": ([("v", "greater", ("in0", "in1"))], (x0 > x1).astype(np.float64)),
    "where": ([("c", "greater", ("in0", "in1")), ("v", "where", ("c", "in0", "in1"))], np.where(x0 > x1, x0, x1)),
    "logical_and": ([("p", "greater", ("in0", "in1")), ("q", "less", ("in0", "in1")),
                     ("v", "bitwise_or", ("p", "q"))], (x0 != x1).astype(np.float64)),
}


def statement(result: str, op: str, args) -> str:
    if op == "let":
        return f"Let(name='{result}', value=Literal(value={float(args)}, dtype='float'), dtype={F32})"
    names = ", ".join(f"'{a}'" for a in args) + ("," if len(args) == 1 else "")
    return f"Assign(name='{result}', op=ElementwiseImpl('{op}'), args=({names}), dtype={F32})"


def write_problem(folder, steps) -> None:
    load = "Load(names=('{n}',), input='{b}', index=(Var(name='a0'),), dtype=" + F32 + ")"
    body = [load.format(n="in0", b="x0"), load.format(n="in1", b="x1")]
    body += [statement(*step) for step in steps]
    body.append(f"Write(output='out', index=(Var(name='a0'),), values=('v',), value_dtype={F32}, "
                f"atomic=False, swizzle='NONE')")
    tile = (f"Tile(axes=(Axis(name='a0', extent=Dim({N}), window=None),), body=({', '.join(body)},), "
            "block_threads=None, aux_threads=0, raster_axes=None, raster_group=None, raster_orient='m')")
    tensor = {"shape": [N], "dtype": "f32"}
    ir = {"inputs": ["x0", "x1"], "outputs": ["out"], "nodes": {
        "x0": {"op": "InputOp", "op_fields": {}, "inputs": [], "outputs": [{"name": "x0", **tensor}]},
        "x1": {"op": "InputOp", "op_fields": {}, "inputs": [], "outputs": [{"name": "x1", **tensor}]},
        "out": {"op": "KernelOp", "op_fields": {"body": [tile], "name": "k_op", "zero_delegated": [], "serial": []},
                "inputs": ["x0", "x1"], "outputs": [{"name": "out", **tensor}]}}}
    kernel = {"node": "out", "entry": "k_op", "arg_order": ["x0", "x1", "out"], "grid": [N // 256, 1, 1],
              "block": [256, 1, 1], "smem_bytes": 0, "zero_outputs": [],
              "args": [{"name": "x0", "dtype": "f32", "shape": [N], "seed": 0},
                       {"name": "x1", "dtype": "f32", "shape": [N], "seed": 1},
                       {"name": "out", "dtype": "f32", "shape": [N], "output": True}]}
    (folder / "k_op.ir.json").write_text(json.dumps(ir))
    (folder / "meta.json").write_text(json.dumps({"kernels": [kernel]}))


def run_on_gpu(ptx: str, kernel: dict) -> np.ndarray:
    compiled = post(RUNNER, "/compile", {"ptx": ptx})
    assert compiled.get("ok"), compiled.get("stderr")
    run = {"cubin_id": compiled["cubin_id"], "entry": kernel["entry"], "grid": kernel["grid"],
           "block": kernel["block"], "smem_bytes": 0, "zero_outputs": [], "args": kernel["args"],
           "warmup": 0, "iters": 1, "max_return": N}
    result = post(RUNNER, "/run", run)
    assert result.get("ok"), result
    return np.array(result["outputs"]["out"]["values"], dtype=np.float64)


@pytest.mark.parametrize("name", CASES)
def test_op_matches_numpy_on_the_gpu(name, tmp_path):
    steps, expected = CASES[name]
    write_problem(tmp_path, steps)
    kernel = json.loads((tmp_path / "meta.json").read_text())["kernels"][0]
    got = run_on_gpu(compile_file(tmp_path / "k_op.ir.json"), kernel)
    expected = expected.astype(np.float32).astype(np.float64)
    abs_err = np.abs(got - expected)
    rel_err = abs_err / np.maximum(np.abs(expected), 1e-30)
    print(f"\n{name:18} max abs {abs_err.max():.2e}  max rel {rel_err[np.abs(expected) > 1e-6].max():.2e}")
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)
