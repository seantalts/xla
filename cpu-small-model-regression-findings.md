# XLA:CPU small-model regression — profiled root-cause map (jax-ml/jax #26145, #37465, #33666)

**Author:** seantalts · **Date:** 2026-06-22 · **Status:** findings, pre-implementation

## TL;DR

The three reported "CPU got slower with the thunk runtime" issues have **three different root causes**, and **none is thunk fragmentation** (the premise behind the Stage-0 region-hoisting pass). Each was confirmed by profiling current-main XLA against jax 0.4.30 vs 0.10.1; several initial hypotheses were measured wrong and discarded. None is a contained one-patch fix — each needs its own scoped work.

| issue | workload | clean regression | root cause | fix locus | difficulty | bounded win |
|---|---|---|---|---|---|---|
| **#37465** | `jnp.diff` regularization | cost 9.4→25µs (2.7×) | scalar **reduce-window** emission | experimental tiling stack (→ then vectorizes free) | feature | ~12µs / **33%** of cost |
| **#26145** | jaxley branched-cable sim | 2.5–4.2× | per-iteration **dispatch** inside while-loop (~28 kernels/step), **gated by scatter** | MLIR region emission (Stage-1) | deep | **~2–4×** |
| **#33666** | diffrax stiff ODE | 1.54× (stiff only) | LAPACK FFI + python callback in Newton loop | FFI/callback overhead | n/a for hoisting | ~0 from hoisting |

## Method & discarded hypotheses

Measured via clean jit'd fixed-input timings (`venv`=0.10.1, `venv-old`=0.4.30 in `perf-regression/`), per-kernel attribution via locally-built `bench_hlo --profile` on current-main XLA and `jax.profiler` perfetto traces, plus LLVM-IR dumps. Hypotheses that profiling **refuted**:

- "bench_hlo can't see the win / it's JAX-side dispatch" — **false**; bench_hlo reproduces the XLA-side regression (mm 70µs ≈ JAX 63µs).
- "scalar-tanh vectorization regression" — **false**; that was an artifact of routing frag through Stage-0's legacy `ComputationKernelEmitter`. The shipping **fusion** emitter vectorizes frag fine (640 vector ops, 0 scalar fabs).
- "thunk fragmentation is the bottleneck" — **false for the proxies** (frag is dispatched once; collapsing 59→1 thunk = 0 delta. mm exceeds the 64KB region gate and never hoists). **True in spirit only for the loop-structured issues**, where dispatch is *per loop iteration* — but those are gated (below).

Measurement traps that bit us, recorded for next time: eager ops inside timed loops (`frag_repro.py`'s `run(v0+1e-9*k)` inflated 7.6µs→27µs); `bench_hlo`'s ~12µs PjRt per-execute floor masking tiny workloads (noop=13µs); and dumping the wrong emitter path.

## Per-issue detail

### #37465 — `jnp.diff` regularization → scalar reduce-window

- `jnp.sum(diff*diff)` over a 64×64 lowers (in **both** jax versions) to `reduce-window {size=32x32 stride=32x32 → f32[2,2]}` then a 2×2→scalar reduce. The lowering is not new; the **emission quality** regressed.
- Forward `cost` is dominated by 4× `wrapped_reduce-window` @ ~3.1µs = ~12µs of ~36µs. Those kernels emit **scalar** (0 vector ops, nested scalar loops) via legacy `xla/service/cpu/ir_emitter.cc:747 HandleReduceWindow → DefaultAction`.
- **Component decomposition (the disambiguating evidence):** `sum(64×64)→scalar` regressed 2.33→12.33µs (+10µs); the per-call host floor only +1.3µs. A vectorized full reduce of the same data is ~0.7µs device → **~10–12µs recoverable**.
- **Fix locus (empirically verified, not the obvious guess):** the cure is *not* a change to `vectorized_reduce_emitter.cc`. Once a non-overlapping reduce-window is lowered to `stablehlo.reduce`, `shlo_to_vector` + `vectorized_reduce_emitter` vectorize it for free. Getting it there requires 4 layers in the **experimental / GPU-shared tiling stack**:
  1. `xla/codegen/tiling/experimental/tiling_space.cc` — `ProcessReduceWindow` (create sequential window dims; today no-op).
  2. `xla/codegen/tiling/experimental/tile_propagation.cc` — windowed affine map `in = out*stride + w − pad_low` (exists only in legacy `xla/service/.../indexing_analysis.cc:782 ComposeIndexingMapsForWindow`).
  3. `xla/codegen/xtile/codegen/experimental_fusion_emitter.cc` — `EmitReduceWindow` (decompose to masked `stablehlo.reduce`, padding → init value).
  4. `xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.cc:177 IsSupportedInstruction` — add `kReduceWindow` (trivial, last).
  Probe confirmed: flipping only the gate → `UNIMPLEMENTED EmitReduceWindow`, and the tiling space mis-tiles reduce-window as pass-through (no reduce/window dims).
- **Unexplored cheaper alternatives** (worth a short spike before committing to the 4-layer feature):
  - (a) Vectorize the legacy `HandleReduceWindow` DefaultAction directly for the non-overlapping case (architecturally backwards, but where the code runs today).
  - (b) Targeted HLO rewrite: non-overlapping reduce-window (`size==stride`) → reshape-to-tiles + `reduce`, inserted **after** the tiling rewrite so it isn't reasserted (naive `jnp.sum` de-tiling *is* rewritten back).
- grad path (1.7×) is separate (pad/transpose/elementwise), lower priority.

### #26145 — jaxley → per-iteration dispatch, gated by scatter

- Real `jaxley` 0.5.0 branched cable, `bwd_euler`, 300 steps. 2.5–4.2× regression (reporter said 5–10×). ~28 distinct tiny kernels fire per timestep; the branched sparse solve is expressed as **scatter (987/119) + gather (705/132)**, not triangular-solve.
- **Win is ~2–4×, not 13×.** The `ThunkExecutor::Execute` "422ms vs 31ms compute" figure is profiler-inflated (recursive Execute + wait-for-completion accumulation). The removable part is intra-step scheduling (~28→1 kernel/step), **not** the loop trip count — the while-thunk driver, one buffer-table setup, and one call/step remain. Anyone claiming >5× is overclaiming.
- **Blocker CONFIRMED:** the while body transitively contains `kScatter`, which is on `InstructionIsUnavailable`, so `SmallWhileLoopHoistingPass` (and the generalized `SmallRegionHoistingPass`) refuse to hoist it. Verified: `xla_cpu_small_call` count = 0 on `module_0272.jit_run`, while a scatter-free sibling module *does* hoist.
- **Unblock is Stage-1, not a legacy hack.** `IrEmitter::HandleScatter` (`ir_emitter.cc:1896`) is `Unimplemented("Scatter is not implemented on CPUs")`, and the legacy `ComputationKernelEmitter` path (which Stage-0 uses) has no scatter generator. But scatter *does* run in jaxley today — via the **MLIR fusion emitter**. So the fix is to back region emission with the MLIR fusion-emitter stack (which already supports scatter) = **Stage-1 region compilation** from the region-compilation design doc. The design's staging anticipated exactly this gap (Stage-0/legacy can't do scatter; Stage-1/MLIR can).

### #33666 — diffrax → FFI/callback overhead (not a hoisting target)

- Regression is **only on the stiff path** (Kvaerno5 1.54×; non-stiff Tsit5 is *not* regressed, actually faster on 0.10.1). Matches the reporter's "stiff ~2×."
- Dominated by LAPACK FFI (`lapack_sgetrf_ffi` + 2× `lapack_strsm_ffi`) and a python callback **inside** the Newton while (`module_0014`, every Newton iteration). These need runtime services (`allow_runtime_calls=false` in the kernel emitter forbids them) — they fundamentally cannot be hoisted into one kernel.
- **No-go for loop-body hoisting.** This is an FFI/callback dispatch-overhead problem (8×8 LU+trsm per Newton step) — a separate workstream from the small-model codegen/region work.

## Stage-0 region-hoisting pass — status

`SmallRegionHoistingPass` (`xla/service/cpu/small_region_hoisting_pass.{h,cc}` + test, branch `feat/cpu-small-region-hoisting`, worktree `xla-defrag`) is **built, correct, 11/11 unit tests GREEN** including bug-shaped guards for all three issues. It generalizes the while-only hoister to maximal region partitioning with liveness outlining. **But it targets thunk fragmentation, which is not the bottleneck for any of these three issues**, and its `ComputationKernelEmitter` substrate can't express the scatter that jaxley needs. Keep it as the **region substrate** for Stage-1; do not ship it as a fix for these issues.

## Recommended tracks (each its own scoped effort)

1. **#37465 reduce-window** — spike the two cheaper alternatives first; if neither lands cleanly, it's a reduce-window feature in the experimental tiling stack (design-doc item, cross-backend blast radius). Confirmed ~33% prize.
2. **#26145 jaxley → Stage-1 region emission** — the evidence-justified version of the defrag thesis: region compilation on the MLIR fusion-emitter stack, which handles scatter. Helps the loop-structured small-model class broadly. ~2–4×.
3. **#33666 diffrax** — defer to an FFI/callback-dispatch workstream; not addressable by region hoisting.

## Reproduce

- Repros: `/tmp/iss37465.py`; `perf-regression/{jaxley_repro.py,diffrax_repro.py}` (jaxley pinned 0.5.0 + `save_exp` shim — does not change the graph); HLO dumps in `perf-regression/{jaxley-dump,diffrax-dump}/`.
- bench: `/Users/xitrium/claud/xla-defrag/bazel-bin/xla/tools/bench_hlo --hlo_file=… --warmup --iters --profile` (use `--run_hlo_passes=false` on `after_optimizations` dumps).
- Worktrees: `xla-defrag` (Stage-0 pass + bench harness), `xla-reducewin` (bench harness only; reduce-window probe reverted, **no fix committed**).
