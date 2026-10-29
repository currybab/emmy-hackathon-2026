# ptxc: design and measurements

This report explains how `compile.py` turns Emmy Kernel IR into PTX, why each part looks the
way it does, and what was measured along the way, including the experiments that did not pay
off. [README.md](README.md) has the summary.

All timings are medians of 100 launches after 10 warm-up launches on the organizers' runner
(RTX 5090), taken with `scripts/bench.py`, unless they are marked as single launches. Compile
times and SASS listings come from an RTX 5090 machine with CUDA 13.3 (`ptxas -v`, `cuobjdump
-sass`). The nvcc comparison is each problem's reference `.ref.cu`, compiled by nvcc to PTX for
sm_89 and run under the same launch.

## 1. Pipeline

```
loader.py      IR json -> the KernelOp's Tile; meta.json -> params (in arg_order), grid, block
compiler.py    prologue -> Lowerer(body) -> vectorize pass -> render
lower.py       one method per statement kind, plus cooperative.py and async_copy.py
builder.py     fresh virtual registers, labels, the .reg declarations, the module text
```

The PTX uses fresh virtual registers (`%f1, %f2, ...`) and leaves register allocation to
ptxas, as the task suggests. Every IR statement is written as a comment above its
instructions. Anything unsupported raises `CompileError`, so the failure is a clear message
rather than PTX that assembles but computes the wrong thing.

### Prologue

- **Parameters.** One `.u64` per buffer, in meta.json's `arg_order`, converted with
  `cvta.to.global`. The IR alone does not fix that order; meta.json is the contract.
- **Thread id.** `gid = ctaid.x * ntid.x + tid.x`.
- **Tile decode.** Row-major: the last axis varies fastest, and each axis is `(gid / stride) %
  extent`. A Tile with a `raster_group` reorders its blocks with the standard threadblock
  swizzle: G block-rows of the grouped axis come before the other axis advances, so
  neighbouring blocks share operand tiles in L2, and a ragged last group is handled. A small
  interpreter in `tests/test_raster.py` runs the emitted integer PTX for every gid and checks
  the order.
- **Guard.** `gid < N` exists only when the launch has more threads than the Tile has cells.
  Only pointwise Tiles are over-launched, so the extra threads can exit early: no barrier
  follows.

### Statements

| IR | PTX |
|---|---|
| Load / Write | `ld/st.global.f32`; a grouped Load or Write of 2 or 4 values becomes `.v2` / `.v4` |
| Write (atomic) | `red.global.add.f32`: an atomic with no result read back |
| Assign | elementwise.py / transcendental.py |
| Accum | updates its register in place; seeded with the op's identity before the loop that folds it |
| Let / Init | `mov.f32` of the constant |
| StridedLoop / Loop | a top-tested loop, or unrolled copies (section 3.2) |
| Cond | a predicate and a branch; a predicate known at compile time keeps one side only |
| Smem | a module-scope `.shared` array, or a slice of the dynamic pool beyond 48 KB |
| Sync | `bar.sync id[, count]`, or `bar.warp.sync` |
| WarpShuffle | XOR butterfly with `shfl.sync.bfly.b32`, strides 16, 8, 4, 2, 1 |
| TreeHalve | halving tree in shared memory with a barrier per level, then everyone reads slot 0 |
| CpAsyncCopy / Commit / Wait | `cp.async.cg` (16 B) or `.ca` (4, 8 B), `commit_group`, `wait_group n` |

WarpShuffle and TreeHalve do not assume `+`. At every level they bind the partner's state and
lower the IR's own `combine_states`. The same code therefore reduces a sum (r01), a max, and
online softmax's coupled (max, sum) pair (f02, a01).

### Index arithmetic (expr.py)

An integer value is either a Python int, known at compile time, or an s32 register. Constant
subtrees fold away: `mul_c1[0]` needs no index math at all. Memory offsets are kept one step
further, as **(variable, constant)** pairs; section 3.1 explains why.

## 2. Numerics

**FMA.** `add.f32` and `mul.f32` carry no rounding modifier, so ptxas may contract a multiply
that feeds an add into one FFMA. That is nvcc's default too, and the SASS shows FFMA. The first
version wrote `.rn` everywhere, which forbids contraction. `div`, `sqrt` and `rcp` keep `.rn`,
because f32 requires a rounding mode on them.

**Constants** are written as exact hex literals (`0f3F800000`), so no decimal-to-binary
rounding happens in ptxas.

**exp.** PTX only has `ex2.approx`, which computes 2^x. The first version, `ex2(x·log2 e)`,
loses accuracy as |x| grows, because the rounding error of `x·log2 e` is scaled by |x|. The
current version:

1. Clamp x to [-104, 89], beyond which e^x is 0 or inf in f32. `max.NaN` keeps NaN.
2. Round t = x·log2 e to an integer i by adding 1.5·2²³. The sum's low mantissa bits hold i,
   so i is available both as a float and, after one integer subtraction, as an s32. There is
   no `cvt`; conversions run at a quarter of the ALU rate.
3. r = x·log2e − i with log2 e split as hi + lo (two FMAs), so |r| ≤ ½ to about 48 bits.
4. 2^r = `ex2.approx.ftz`. Neither r nor 2^r can be subnormal, so `.ftz` is safe and removes
   the subnormal handling.
5. 2^i is built directly in the exponent field, as 2^h·2^(i−h) so both halves stay normal.
   Overflow to inf and underflow to 0 then happen in the final multiplies.

The GPU result against numpy in float64 has a max relative error of 1.2e-7 (about 1 ulp),
including inputs up to ±100, and overflow gives exactly inf.

| version of exp | f02 time | a01 time |
|---|---|---|
| `ex2(x·log2e)` (inaccurate) | 2.98 µs | 64 µs |
| range-reduced, with `cvt` and two `ex2` | 4.64 µs | 74–89 µs |
| magic-number rounding, 2^i from exponent bits | 4.70 µs | 74.5 µs |
| + `.ftz` on the one remaining `ex2` | **2.98 µs** (nvcc 2.98) | 67 µs |

The other transcendental ops are built the same way. Their max absolute error against numpy
over 1024 standard-normal inputs:

| op | how | max abs error |
|---|---|---|
| log | `lg2.approx` · ln 2 | 9.5e-7 |
| sigmoid | 1 / (1 + e^-x), Emmy's own expansion | 6e-8 |
| tanh | sign(x)·(1 − 2/(e^{2\|x\|}+1)), which cannot overflow | 1.0e-7 |
| erf | Abramowitz & Stegun 7.1.26 | 3.2e-7 |
| sin, cos | reduction by 2π (hi + lo), then `sin/cos.approx` | 4.3e-7 at \|x\| ≤ 40 |
| pow | 2^(y·log2\|x\|) with C's sign and zero rules | rel. 8.9e-7 |
| comparisons, where, min, abs, neg | `setp` / `selp`, C's bool ↔ float rules (`!=` is true for NaN) | 0 |

## 3. Performance

The performance work followed one loop: measure against nvcc, read both SASS listings, find
the instruction-level difference, fix it in the compiler, and measure again. Three experiments
made things worse, so they were reverted or narrowed; they are listed too.

### 3.1 Integer work: m01 matmul, 2.8× → level

In the first matmul, every one of the 64 unrolled k-steps recomputed its shared-memory address:

```
div.s32 %r293, %r152, 64      // _ks / 64, again for every k
rem.s32 %r294, %r293, 3
mul.lo.s32 %r295, %r294, 64
add.s32 %r296, %r295, 1       // + k  (the constant ends up inside the multiply)
mul.lo.s32 %r297, %r296, 64
```

The PTX had about 3,800 integer instructions against nvcc's 60. The SASS had 262 `IABS`, from
signed-division sequences, and nvcc's had none. nvcc writes `[%r98+256]`: the constant goes
into the address. **ptxas cannot make that rewrite.** PTX `add.s32` wraps, and the index is
sign-extended to 64 bits by `mul.wide.s32`, so `4·(t+1)` is not provably `4·t + 4`. nvcc's
front end can make the rewrite because C's signed overflow is undefined. The IR → PTX compiler
has to do it, so expr.py now does three things:

1. **(variable, constant) offsets.** `(t + 1) * 64` lowers to the pair `(t·64, 64)`, and the
   constant becomes the `[reg+imm]` immediate.
2. **Unsigned division when the dividend is provably non-negative.** Axes, loop counters,
   lane and warp start the proof, and `+`, `*`, `/` and `%` preserve it. `/ 64` then becomes a
   shift, and the signed sequence disappears.
3. **Reuse of computed integer values and addresses.** Values computed inside a branch, a loop
   body or a TreeHalve level are dropped on exit, because some threads skip them or they
   change every iteration.

Single-launch times went 77.5 → 73 µs (step 1) → 32 µs (steps 2 and 3). In warm timing that
was 20.0 µs against nvcc's 15.9, or 1.26×. The rest came from **vector loads** (section 3.4):
**15.7 µs**.

### 3.2 Unrolling: r02/r03 1.9× → level, a01 ptxas 5.5 s → 54 ms

nvcc's front end unrolls loops on its own judgment. Here the compiler makes that judgment
(lower.py):

- **Strided loops** with data-only bodies (Load, Assign, Accum, Let, Write) run several
  iterations per trip, then the remainder one by one. Copy j sees the axis as `(counter,
  j·step)`, so its constant becomes an address immediate. The iterations stay in order, so the
  results are bit-identical.
- **Small Loops** are copied out whole, up to 256 statements, even without an IR request.
- **An unroll the IR asks for is declined** when the copies would exceed 1024 statements,
  counting nested loops at their full extent.

| step | r02_big_sum | evidence |
|---|---|---|
| no unrolling | 21.0 µs (1.95×) | one load in flight per thread; nvcc SASS has 29 LDG, ours 7 |
| 4 iterations per trip | 23.1 µs, **no gain** | nvcc also unrolls by 4, but ptxas unrolls nvcc's counted loop 4× again: 16 loads in flight. Our `counter + 3·step < end` loop is not re-unrolled. |
| up to 16 per trip, sized by body and trip count | **10.8 µs** (nvcc 10.8) | |

a01's outer loop has `unroll=True`, but its body holds four 64-step loops. Following the
request produced 33,000 lines of PTX and **5.5 s of ptxas**. nvcc ignores that pragma too: its
PTX keeps the loop. With the 1024-statement budget the request is declined, and the inner loops
are unrolled instead. The 15 kernels now take 0.21 s of ptxas in total, and a01 runs in 60 µs,
faster than it ran fully unrolled.

### 3.3 Loop-invariant code motion: a01 60 → 57.9 µs

a02's loop over 128 keys re-read the same row of q in every iteration: 65 loads per iteration
against nvcc's 33. licm.py moves an instruction in front of a rolled loop when all of these hold:

- it is pure, or it is a global load in a Loop that writes no global memory;
- all its inputs come from outside the loop or from already-moved instructions;
- its result is defined once and not read before that definition;
- it is not inside a branch or an inner loop.

Loads only move out of a `Loop`, which runs its static extent of at least one. A strided loop
may run zero times, and a hoisted load would then be an access the program never made.

The loop now has 33 loads per iteration, like nvcc. a02's time did not change, so ptxas had
already been hoisting most of them. a01 went from 60 to 57.9 µs.

### 3.4 Vector shared loads: m01 20.7 → 15.7 µs

nvcc reads matmul tiles with `ld.shared.v2` and `.v4`: 102 LDS against our 166 scalar ones.
vectorize.py merges neighbouring scalar loads after the body is emitted, which needs two proofs:

- **Alignment.** Every s32 value carries a factor it is known to be a multiple of: `+` and `−`
  take the gcd, `*` the product. Every address register records its byte alignment, combining
  that factor with the base's alignment. `a_smem[row, k]` rows are 64-element multiples, so
  the address is 256-byte aligned and four k-steps form one `v4`. `b_smem[k, a1_u·2]` is only
  8-byte aligned, so it gets `v2`.
- **Nothing in between writes.** Merging happens only within stretches without a store,
  atomic, barrier, cp.async, branch or label. Load destinations are always fresh registers, so
  defining them earlier is safe.

m01 now has 64 `v2` and 32 `v4` loads, the same as nvcc, and runs in 15.7 µs (nvcc 15.9).

### 3.5 Experiments that made things worse

| experiment | result | decision |
|---|---|---|
| merge **global** loads into vectors too | a01 60 → 72 µs, a02 33 → 37 µs | shared loads only |
| strength reduction (a pointer bumped each iteration instead of `base + 4·counter`) | a02 33.3 → 35.4 µs, the same in repeated runs | reverted |
| unroll strided loops by 4 | r02 21.0 → 23.1 µs | raised to 16 (section 3.2) |

Why the first two hurt is not established. The likely cause is longer live ranges, or a
different ptxas schedule, when work moves to the top of a loop.

## 4. Where it stands

| | |
|---|---|
| open problems | 15 / 15 on the runner |
| kernel time vs nvcc | 14 level or faster (a01 22% faster), a02 14% slower |
| ptxas time, 15 kernels | 0.21 s (nvcc path: 2.2 s) |
| tests | 155 unit tests; 35 on the runner (15 problems, 20 op checks) |
| code | ptxc/ ≈ 1,800 lines in 13 modules |

**Known gaps**

- **a02 is 14% slower.** Its online-softmax recurrence runs on one warp per block, 128 times
  in sequence. Our exp spends two `max/min` instructions on the clamp and two multiplies on
  2^i; that precision for extreme inputs sits on the critical path.
- **tanh and erf near 0.** The relative error is large because 1 − (almost 1) cancels, but the
  absolute error stays below 1.5e-7.
- **Not supported**, and rejected with an error: non-f32 buffers, launches that are not 1-D,
  swizzled cp.async, and the tensor-core and TMA statements. TASK.md marks the last ones as
  appearing in no problem.
- a01 uses 255 registers, as nvcc's build of it does.
