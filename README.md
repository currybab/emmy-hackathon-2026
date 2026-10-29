# ptxc: Emmy Kernel IR → PTX

`compile.py` translates one Emmy Kernel IR kernel straight to PTX, with no CUDA C++ and no
nvcc in between:

```bash
python compile.py problems/r01_array_sum/k_sum_1_reduce.ir.json > out.ptx
```

It is written for the [Emmy Hackathon 2026](docs/HACKATHON.md). The task statement is in
[TASK.md](TASK.md). This README is the overview; [REPORT.md](REPORT.md) explains the design
decisions and the measurements behind them.

## Results

**Correctness.** All 15 open problems pass on the runner (RTX 5090), across all five
categories.

| category | problems | max abs error |
|---|---|---|
| 1 pointwise | p00–p03 | 0 (bit-exact) |
| 2 reductions | r01–r04 | 9e-10 – 1.5e-5 |
| 3 matmul | m01, m02 | 8e-5, 1.1e-5 |
| 4 fused | f01–f03 | 7e-9 – 1.9e-6 |
| 5 attention-like | a01, a02 | 6.6e-7, 3.9e-7 |

The compiler also lowers every f32 elementwise op Emmy's CUDA renderer can emit, 34 in all.
Twenty GPU checks compare the transcendental ops and representative comparisons and selects
with numpy in float64; exp stays within about 1 ulp over the whole f32 range.

**Compile time.** The 15 kernels take 0.21 s of `ptxas` in total. Emmy's own path, nvcc on the
generated CUDA C++, takes 2.2 s for the same kernels. Both were timed on an RTX 5090 box with
CUDA 13.3, best of 3. `compile.py` itself takes about 130 ms per kernel, and about 107 ms of
that is the Python start-up and `import emmy`.

**Kernel time.** Median of 100 launches after 10 warm-up launches on the runner, compared with
the nvcc build of each problem's reference CUDA:

| problem | ours (µs) | nvcc (µs) | |
|---|---|---|---|
| a01_attention | 57.9 | 74.3 | **22% faster** |
| a02_attention_128 | 33.3 | 29.3 | 14% slower |
| m01_matmul_512 | 15.7 | 15.9 | level |
| r02_big_sum | 10.8 | 10.8 | level |
| r03_mean | 6.7 | 6.7 | level |
| the other 10 | 3.0 – 4.9 | 3.0 – 5.0 | level (±1%) |
| **sum of all 15** | **158** | **171** | |

## How it works

```
*.ir.json + meta.json
   │  loader.py       Graph.from_dict, the single KernelOp, params in meta.json arg_order
   ▼
compiler.py           prologue: params, gid, Tile axis decode (with block rasterization), guard
   │
lower.py              one handler per IR statement; loops, unrolling, shared memory
 ├ expr.py            integer index math: constant folding, (variable, constant) offsets,
 │                    reuse of computed values, non-negativity and alignment facts
 ├ elementwise.py     f32 ops that are a single instruction
 ├ transcendental.py  exp, log, tanh, erf, sigmoid, sin, cos, pow, comparisons, where
 ├ cooperative.py     Sync, WarpShuffle (shfl.bfly butterfly), TreeHalve (shared-memory tree)
 ├ async_copy.py      cp.async copy / commit / wait
 └ licm.py            moves loop-invariant work in front of rolled loops
   │
vectorize.py          merges neighbouring shared loads into ld.shared.v2 / .v4
   │
builder.py            virtual registers, labels, module text  →  PTX on stdout
```

Every IR statement is written into the PTX as a comment above the instructions it became, so
the output reads side by side with the IR:

```
	// f32 acc0 <- add(acc0, in0)
	add.f32 %f1, %f1, %f2;
```

Errors are loud on purpose. Anything the compiler does not handle stops with
`compile.py: error: ...` and exit code 1, rather than emitting PTX that assembles but computes
the wrong thing.

## Design highlights

Each of these is explained, with its measurement, in [REPORT.md](REPORT.md).

- **Combines are generic.** WarpShuffle and TreeHalve run the IR's own `combine_states`, so
  the same code reduces a sum, a max, or online softmax's (max, sum) pair.
- **Index constants become address immediates.** PTX integer adds wrap, and indices are
  sign-extended to 64 bits, so ptxas may not rewrite `4*(t+1)` as `4*t + 4`. The compiler keeps
  offsets as (variable, constant) and does the rewrite itself. Combined with unsigned division
  on indices proven non-negative and reuse of computed index math, this took m01 matmul from
  2.8× slower than nvcc to 1.26×. Merging shared loads (below) closed the rest.
- **The compiler decides unrolling, as nvcc's front end does.** Strided reduction loops run up
  to 16 iterations per trip, so each thread keeps 16 loads in flight. An unroll the IR asks for
  is declined when the copies would exceed 1024 statements. On a01, accepting it cost 5.5 s of
  ptxas; declining it brought that down to 54 ms and the kernel ran faster.
- **exp is accurate and short.** It is range-reduced around an integer power of two, with
  log2(e) split into hi and lo parts. The integer part is rounded with the 1.5·2²³ trick, and
  2^i is built from exponent bits. That leaves one `ex2.approx.ftz` and no conversion
  instructions.
- **Vector loads need an alignment proof.** Every index carries a known factor, and every
  address register carries its byte alignment. Neighbouring shared loads are merged into v2 or
  v4 only when that alignment is proven.

## Running the tests

```bash
pip install emmy-ml==0.3.11 numpy requests pytest
python -m pytest                     # 155 unit tests, no GPU needed
PTXC_REMOTE=1 python -m pytest       # + the 15 problems and 20 op checks on the runner
python scripts/bench.py              # kernel times on the runner
```

`tests/irkit.py` builds Kernel IR from Emmy's real classes, so each lowering rule is tested on
hand-made IR. `tests/test_ops_remote.py` builds a one-op kernel per elementwise op and compares
the GPU result with numpy.

## Credits

- tinygrad's PTX renderer: the idea of writing f32 constants as exact hex literals (`0f3F800000`).
- `examples/k_sum_1_reduce.ptx` from the organizers: the starting point for the reduction lowering.
- Abramowitz & Stegun, formula 7.1.26, for erf.
- Emmy's source was read to learn the IR's semantics, such as the Tile decode, raster order, the
  dynamic shared-memory pool and the op set. None of its rendering code is used. The only Emmy
  call is `Graph.from_dict`, to load the IR.
- nvcc's output was used only as a measuring stick: register counts, SASS instruction mixes and
  timings to compare against.
