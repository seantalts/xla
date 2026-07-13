# XLA:CPU performance on small scientific computing models

**Author:** seantalts
**Date:** 2026-07-09
**Status:** Draft for review
**Code:** [`feat/cpu-small-region-hoisting`](https://github.com/seantalts/xla/tree/feat/cpu-small-region-hoisting)
**Measurement record:** [`notes/cpu-small-model-regression-findings`](https://github.com/seantalts/xla/tree/notes/cpu-small-model-regression-findings)

## Summary

Small scientific computing models (ODE steppers, neuron simulators, physics engines) run 2x to 10x slower since XLA:CPU moved to the thunk runtime. The dominant cost is not the code inside any kernel. It is the number of kernels: these models execute tens of thousands of tiny kernels per step, at roughly 50 ns of dispatch and call overhead each, where the old runtime compiled the whole program into one function and spent roughly 7 ns per op.

Two passes already on the branch address the worst reported case. Region hoisting folds runs of small-footprint instructions, loops included, into single kernels. Small-scatter expansion rewrites small scatters into loop form so they stop splitting those runs. On [jax#26145](https://github.com/jax-ml/jax/issues/26145) (jaxley) these take a step from 1.53 ms to 0.36 ms with bitwise-identical output, at parity with the pre-thunk runtime.

Those two passes are one milestone in a larger plan. This doc lays out that plan: treat whole-region compilation as a first-class execution regime next to thunk-granular execution, then grow what a region can contain (runtime calls, milestone M2) and how well it compiles (a modern region emitter, M3), and stop generating pathological input in the first place (loop structure, M4). Each reported issue maps to a milestone below; fixes for [jax#37465](https://github.com/jax-ml/jax/issues/37465), [jax#33666](https://github.com/jax-ml/jax/issues/33666), and [jax#26021](https://github.com/jax-ml/jax/issues/26021) are scoped individually.

## Background

### The workload class

The affected models share a shape: a stepping or solver loop that runs hundreds to thousands of iterations per user-visible call, over arrays of tens to a few thousand elements. Per-iteration math is small: 4x4 solves, segment sums, elementwise updates on short vectors. The reporters are neuron simulation (jaxley), stiff ODE integration (diffrax), rigid-body kinematics (MuJoCo MJX), and plain `jnp.diff` regularization terms.

### What changed in the runtime

The legacy XLA:CPU runtime compiled the entire entry computation into one LLVM function. While loops were native loops; values flowed between ops inside one optimized function. The thunk runtime emits one kernel per instruction or fusion and executes a dependency graph of thunks. For large models this is the right design (parallelism, profiling, library integration). For the class above it replaces ~7 ns/op straight-line execution with ~50 ns per kernel execution: ~18 ns of dispatch (measured with a calibrated slope test) plus ~30 ns of kernel ABI and prologue work at these array sizes.

## The reported problems

| issue | workload | regression | measured root cause |
|---|---|---|---|
| [jax#26145](https://github.com/jax-ml/jax/issues/26145) | jaxley neuron sim | 4-5x/step | thunk count: a 300-iteration while loop at ~73 thunks per iteration, with scatters splitting foldable runs |
| [jax#37465](https://github.com/jax-ml/jax/issues/37465) | `jnp.diff` regularization | 1.7-3.2x | four scalar-emitted reduce-window kernels (~12 us of ~30 us); see the YNN analysis below |
| [jax#33666](https://github.com/jax-ml/jax/issues/33666) | diffrax stiff ODE | ~1.5-2x | LAPACK FFI calls (getrf, trsm) and a Python callback inside the Newton loop; FFI cannot cross into kernels today |
| [jax#26021](https://github.com/jax-ml/jax/issues/26021) | MJX mass matrix | 3.7x | trace-time-unrolled 35-joint loops produce ~208 dynamic-update-slice fusions; DUS-heavy; root cause partially unconfirmed |
| TORAX (no public issue) | plasma transport | compile ~3 s | the call inliner flattens ~1087 repeated call sites into a 29k-instruction module that the simplification fixpoint re-sweeps |

## Diagnosis

Each conclusion below came from a measurement, after several early inferences proved wrong (Appendix A.6 records the misses).

**jax#26145.** A census of the optimized module gives ~22,400 kernel executions per step for 1.109 ms of wall time, ~50 ns each. An IR audit of the hottest region kernels found mediocre code (unforwarded loads, scalar loops), but hand-merging fusions inside kernels moved end-to-end time by ~1% (Appendix A.3), so kernel quality is not the cost at these sizes. Replacing scatters with placeholders let the region pass fold the entire program into one kernel at 0.238 ms (Appendix A.5), which located the rest of the win behind scatter boundaries.

**jax#37465.** The `jnp.sum(d*d)` reductions get split by the tree-reduction rewriter into reduce-window(32x32, stride 32x32) plus a small reduce, and the reduce-window kernels are emitted as scalar loops (~3.2 us each, four of them). Re-checked at head this week because [openxla/xla@5f26078f](https://github.com/openxla/xla/commit/5f26078f7cb2c6b8872c87e43fe6906923396c23) taught YNNPACK reductions to bypass the tree rewriter: the fix does not apply here. YNN only claims ops with at least 4096 elements in some operand or result (`kMinElements` in [`ynn_support.cc`](https://github.com/openxla/xla/blob/main/xla/backends/cpu/ynn_support.cc), `IsInstructionPreferredByYnn`), and the diff outputs are 63x64 = 4032 and 62x64 = 3968. The repro sits just under the gate, so the reduce is declined, the tree rewriter splits it, and the scalar reduce-windows remain. We are deliberately not touching the YNN gate; the fix stays in XLA's own codegen (see Alternatives). Verified empirically at our base: the compiled module still contains all four reduce-window ops as scalar kernels and zero YNN fusions, at 33 us per call. Region hoisting does not rescue this model either (verified: no region forms, because the 16 KB arrays exceed the summed-bytes gate), and reduce-window would emit scalar inside a region anyway. This issue needs its own fix (see M3 and the fix map).

**jax#33666.** The regression is confined to the stiff solver path. Each Newton iteration issues LAPACK custom calls and a Python callback; these are on the region boundary list, so no folding is possible today. The mechanism to lift this exists in the legacy emitter: `IrEmitter` can emit runtime calls (`allow_runtime_calls`), but region kernels disable it ([`computation_kernel_emitter.cc:248`](https://github.com/seantalts/xla/blob/feat/cpu-small-region-hoisting/xla/backends/cpu/codegen/computation_kernel_emitter.cc)) because their ABI carries no runtime context. M2 below is the plan to thread it.

**jax#26021.** ~208 of 254 kernels are dynamic-update-slice fusions from unrolled per-joint loops. The per-kernel arithmetic is not the cost; the DUS array updates are. Whether the updates copy where the old runtime updated in place is the unconfirmed part; completing that diagnosis is scheduled in the fix map. The arrays are f64 and exceed the small-model gates, so this is a medium-size model: kernel quality and loop structure (M3, M4) are the relevant levers, not dispatch count.

**Loop structure (spike, 2026-07-08).** We compared identical computations in rolled (while/scan) and trace-time-unrolled form, sizes 8 to 512 ([artifacts](https://github.com/seantalts/xla/tree/notes/cpu-small-model-regression-findings/reroll-spike)). On jax 0.4.30 the two forms ran at the same speed and unrolling only cost compile time, which is why unrolling became a habit. On current CPU the unrolled form is 4.5x slower at runtime and ~12x slower to compile. With region hoisting the rolled form wins on both axes at every size (compile time flat at ~13 ms regardless of trip count; ~25 ns per iteration). MJX cannot simply roll its joint loops: per-joint-type dispatch violates `lax.scan`'s shape contract (`mujoco/mjx/_src/scan.py`), so its fix runs through M4 rather than user guidance. The spike also exposed a compile-time hazard in our own pass, discussed under P3.

## Goals

1. Return the affected class to pre-thunk runtime performance (within 10%) with bitwise-identical results, and keep it there with regression tests.
2. No effect on large models: conservative size gates, single-flag escape hatch.
3. A concrete fix path for each reported issue, not only the largest one.
4. Reduce reliance on the legacy emitter over time rather than deepening it.

## Non-goals

- A faster thunk executor. Dispatch is already ~18 ns; the problem is count (Appendix A.1).
- Post-hoc re-rolling of already-unrolled HLO (isomorphic-subgraph detection). Deferred; prevention comes first (M4).
- TORAX compile time. Related (M4 prevention lever) but explicitly out of scope here.

## Plan

### Framing: two execution regimes

XLA:CPU today has one execution regime: thunk-granular kernels. The old runtime effectively had another: whole-program compilation. The plan makes the second regime a deliberate, gated choice again. Per region of the program, the compiler picks between thunk granularity (large tensors, library ops, parallelism) and single-function region compilation (small tensors, loops, sequential math). The milestones then widen what regions can contain and improve how they compile.

### M1: region formation (landed)

- **P1, region hoisting** ([`c710a3a3`](https://github.com/seantalts/xla/commit/c710a3a39d)): generalizes the upstream `SmallWhileLoopHoistingPass`. Partitions the entry computation and all while/conditional bodies into maximal runs of eligible instructions and outlines each run into a call tagged `xla_cpu_small_call`, which the backend emits as one kernel through the existing `ComputationKernelEmitter`. Gates: summed bytes accessed under `max(xla_cpu_small_while_loop_byte_threshold, 64KB)`; at least 4 members or a control-flow op; sentinel 0 disables.
- **P2, small-scatter expansion** ([`aaa4c553`](https://github.com/seantalts/xla/commit/aaa4c553a0)): scatter is a boundary only because the legacy emitter never implemented it. `ScatterExpander(kEliminateAllScatters)` already lowers scatter to a while loop of gather and dynamic-update-slice; we subclass it with a byte-footprint predicate on the same gate and run it at the existing expansion point, before simplification and fusion. Small scatters become foldable control flow; large scatters keep the dedicated MLIR scatter kernel.
- **P3, straight-line region cap** (required, not yet implemented): folding a large straight-line chain into one function inflates compile time superlinearly (measured ~N^1.8: 3x at 32 ops, 11x at 128, 29.7 s force-folded at 512). Flag ablation isolates the cost to LLVM middle-end optimization of one giant basic block: codegen parallelism is irrelevant (split count changes nothing) and opt-level 0 compiles the same function in 73 ms while keeping ~90% of the runtime win. Regions containing loops are unaffected (small basic blocks; jaxley's whole-program fold compiles in 0.34 s). Fix: cap member count for regions with no control flow; optionally compile oversized region functions at a reduced opt level.

### M2: runtime calls inside regions (targets jax#33666)

Let a region kernel call back into the runtime, so custom calls (FFI) and host callbacks stop being boundaries. The legacy emitter already knows how to emit runtime calls; region kernels disable it because the kernel ABI has no runtime context. Work items: thread a run-context pointer through the region kernel ABI (the thunk owns it at execute time); build FFI call frames per call site at emission time (they are static); propagate error status out of the kernel; define threading rules (a region kernel executes on one executor thread; callees that spawn parallelism must be given the device threadpool explicitly). After M2 the boundary list shrinks to infeed/outfeed and collectives, and the diffrax Newton loop folds into one kernel that calls getrf/trsm directly, the same shape the old runtime executed. Risks: re-entrancy and error paths need care; a measurement gate on the actual diffrax repro decides whether FFI dispatch was in fact the cost (the profile says yes, but per our own rules the fix ships with an A/B).

### M3: a modern region emitter (targets jax#37465, jax#26021, and the long term)

Region compilation currently rides on the legacy `IrEmitter`. That is acceptable for M1 (it handles everything, control flow included) but is the wrong long-term base: it emits scalar loops for several ops, it is unmaintained, and every region we route through it deepens the dependency. M3 replaces the region backend with an MLIR emitter: a schedule-walking driver that emits `scf.while`/`scf.if` for control flow, reuses the xtile per-op emitters for op bodies, and keeps intermediates in SSA values and stack allocations. Prior work: the flag-gated tiled region emitter on the branch ([`3a1b7002`](https://github.com/seantalts/xla/commit/3a1b70020e)) proves the routing and the fusion-view mechanics; the coverage findings from the [four-phase plan](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-tiled-region-coverage-design.md) (multi-output roots, reverse as negative stride, DUS in-place hazards, gather) become M3 work items instead of a standalone effort.

Op coverage in M3 is chosen by issue impact rather than generality: reduce-window (vectorized, fixes the #37465 kernel directly and permanently, independent of library size gates), dynamic-update-slice with in-place semantics (the #26021 kernel population), gather/reverse (jaxley-class bodies), scatter as an in-kernel sequential loop (closes the remaining 0.352 vs 0.238 ms gap on jaxley). Honesty about magnitude: at 32-element arrays kernel quality is worth ~1%; the payoff is at medium sizes (#26021's f64 arrays, batched bodies) and in retiring the legacy dependency. M3 also owns compile-time tiering: emit simpler IR and choose opt level by region size, using the O0 measurement above as the anchor.

### M4: loop structure (targets jax#26021 and prevention)

Three levers, in order of cost:
1. **Guidance** (free): with M1, rolled loops are faster to run and to compile than trace-time-unrolled ones. Library and user guidance should say so once the passes ship in a jaxlib release.
2. **Prevention** (small): stop flattening rolled structure we already have. The call inliner turns repeated call sites into unrolled code; keeping small repeated computations as calls (or lowering identical call sites to a loop over stacked operands) preserves the structure M1 can fold. This is also the TORAX compile-time lever (533 call sites collapse to 11 shape signatures), noted here and otherwise out of scope.
3. **Batching for heterogeneity** (upstream, larger): MJX-style per-type dispatch cannot roll as-is; it needs padding plus `lax.switch`, or grouped batching, in the library. Our spike measured the batched form of a homogeneous body at 2-4x over unrolled at width 4, with more available at larger widths. Worth an issue filed against MJX with the numbers rather than XLA-side work.

### Fix map

| issue | fix | milestone | status |
|---|---|---|---|
| [jax#26145](https://github.com/jax-ml/jax/issues/26145) | region hoisting + scatter expansion | M1 | landed on branch, 1.53 -> 0.36 ms bitwise |
| [jax#37465](https://github.com/jax-ml/jax/issues/37465) | near term: rewrite non-overlapping reduce-window (size == stride) to reshape+reduce at the HLO level, handled by XLA's own vectorized reduce emitters. Durable: M3 vectorized reduce-window. A YNN gate change could also catch this case; rejected as a direction (see Alternatives) | M3 (near-term fix independent) | scoped; YNN decline reason verified this week |
| [jax#33666](https://github.com/jax-ml/jax/issues/33666) | runtime calls inside regions; fold the Newton loop, call LAPACK from inside the kernel | M2 | design above; A/B gate on the real repro |
| [jax#26021](https://github.com/jax-ml/jax/issues/26021) | finish the DUS in-place diagnosis (compare update-chain kernels old vs new; confirm or refute the copy hypothesis); then DUS emission fix under M3, plus M4 batching upstream | M3 + M4 | diagnosis task defined; root cause partially unconfirmed |
| TORAX | prevention lever (M4.2) | M4 | out of scope here; census done (533 sites -> 11 shapes) |

## Results to date

Apple M3, single thread, `bench_hlo`, 100 iterations, seed-matched parity. jaxley is [jax#26145](https://github.com/jax-ml/jax/issues/26145) `module_0272`.

| configuration | jaxley ms/step | output |
|---|---|---|
| upstream-equivalent (sentinel 0) | 1.531 | baseline |
| + P1 region hoisting | 1.09-1.12 | bitwise |
| + P2 scatter expansion | 0.354-0.377 | bitwise |
| pre-thunk runtime (jax 0.4.30, end to end) | 0.372 | reference |

After P2 the optimized entry is a single call: one kernel and one thunk per step, with the main loop and 54 expanded scatters as native control flow inside one function.

On generality: P1 and P2 are not jaxley-specific. The rolled-loop microbenchmarks show the same ~3-4x per-iteration overhead reduction on any small rolled loop, and the passes fire on any model under the byte gates. jaxley is simply the only reported issue in the class we can currently verify end to end; #33666 joins it after M2.

Worst case for P2's predicate (a lone small scatter with no neighbors to fold with): 17-18 us vs 16-17 us for the dedicated kernel, because the expanded loop still folds into its own single kernel.

## Testing

- Unit tests: 14+ on P1 (including tests shaped like the reported issues, token threading, control-dependency boundaries), 5 on P2 (small expanded, large untouched, footprint counts all operands and tuple leaves, variadic). P3 adds cap tests.
- Every workload measurement runs seed-matched against the sentinel configuration with a bitwise diff. This caught real problems during development; it stays in the loop.
- Non-regression: a large-scatter benchmark that must not match the predicate; a large-model suite run to confirm the gates keep everything inert; the unrolled-N compile-time matrix from the spike as a P3 regression test.
- M2 and M3 each carry their own A/B gate on the corresponding repro before landing.

## Rollout

1. Land P3, re-run the matrix and jaxley.
2. Upstream in two PRs: P1+P3 (replacing `SmallWhileLoopHoistingPass`, which it strictly generalizes and which is already default-on upstream), then P2. Shared option, shared sentinel, escape hatch documented in the flag description.
3. Default on, matching the while-hoisting precedent. If reviewers prefer a staged flip, the same option supports off-by-default.
4. Comment on the jax issues with reproduction numbers once a jaxlib release carries the passes; update loop guidance (M4.1) at the same time.
5. M2 and M3 are separate design reviews; this doc fixes their scope and gates.

## Monitoring

- Fold-rate counters (already in the region path): regions formed, members, decline reasons; inspectable when a user reports an anomaly.
- The compile-time matrix bounds worst-case inflation; P3 keeps it within noise of upstream.
- Success criteria: jax#26145 closed with maintainer-reproduced numbers; no regressions attributed to the passes for a release cycle; each subsequent issue closed by its mapped milestone.

## Risks

| risk | evidence | mitigation |
|---|---|---|
| compile-time blowup on large straight-line regions | ~N^1.8 measured; 11x at 128 ops | P3 cap; optional reduced opt level (O0 keeps ~90% of the win, 73 ms vs 2.17 s) |
| expanded scatter that never joins a region | +1-2 us measured on the lone-scatter case | conservative gate; expanded loop folds alone; possible v2 predicate that dry-runs region eligibility |
| large-model interference | measured on the in-tree Gemma3 1B benchmark (`gemma3_1b_flax_call`, 1.86 GB of weights): the passes are not inert. 26 regions form on per-layer KV/mask/rope bookkeeping and all 22 tiny attention index scatters expand, yet output is bit-identical, runtime is a wash (164 vs 171 ms), and compile time rises 13% (+95 ms) | 64KB gates bound what can fire; preset gating excludes FAST_COMPILE; re-measure on the deeper in-tree configs (gemma2_2b_keras_jax, gemma4_2b_bf16) before upstreaming |
| numerics | none observed; all results bitwise | parity checks in tests and in the measurement loop; P2 uses XLA's reference scatter lowering |
| M2 re-entrancy and error paths | n/a (design stage) | scope M2 to static FFI call frames first; status propagation designed before code |
| M3 scope creep | four-phase plan history (Appendix A.2) | op coverage strictly issue-driven; measurement gate per op |

## Alternatives considered

Details and numbers in Appendix A.

- Faster dispatch: ceiling ~10%, rejected (A.1).
- Tiled region emission as its own track: built, measured neutral at small sizes, folded into M3 (A.2).
- Merging fusions inside regions: ~1%; 24% slower with duplication (A.3).
- In-kernel scatter codegen: worse than expansion for M1 on every axis; revisit inside M3 (A.4).
- The four-phase tiled coverage plan: superseded by measurement; its findings feed M3 (A.2).
- Widening YNNPACK coverage for small reduce-like ops (jax#37465 misses the `kMinElements` gate by 64 elements, so a one-line relaxation would catch it): rejected as a direction. We prefer to keep small-op performance in XLA's own codegen rather than grow the library dispatch surface, and the YNN size gate exists precisely because library dispatch loses at small sizes.

## Open questions

1. P3 cap value (proposal: 48 straight-line members; compile cost is 3x at 32 in the worst synthetic case, and splitting costs tens of nanoseconds per boundary).
2. Split oversized regions vs compile them at reduced opt level (data says either works; splitting is easier to reason about).
3. M2 context threading: ABI parameter vs thunk-installed thread-local. ABI parameter is cleaner; needs a kernel-signature change.
4. Upstream default-on vs staged flip.

## References

- Issues: [jax#26145](https://github.com/jax-ml/jax/issues/26145), [jax#37465](https://github.com/jax-ml/jax/issues/37465), [jax#33666](https://github.com/jax-ml/jax/issues/33666), [jax#26021](https://github.com/jax-ml/jax/issues/26021)
- Branch: [`feat/cpu-small-region-hoisting`](https://github.com/seantalts/xla/tree/feat/cpu-small-region-hoisting); commits [`c710a3a3`](https://github.com/seantalts/xla/commit/c710a3a39d), [`3a1b7002`](https://github.com/seantalts/xla/commit/3a1b70020e), [`aaa4c553`](https://github.com/seantalts/xla/commit/aaa4c553a0)
- Findings: [root-cause map](https://github.com/seantalts/xla/blob/notes/cpu-small-model-regression-findings/cpu-small-model-regression-findings.md), [decomposition and POC record](https://github.com/seantalts/xla/blob/notes/cpu-small-model-regression-findings/cpu-phase05-dispatch-decomposition-findings.md), [re-rolling spike](https://github.com/seantalts/xla/tree/notes/cpu-small-model-regression-findings/reroll-spike)
- Region emission design docs: [single-kernel region emission](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-stage1-region-compilation-design.md), [tiled coverage (appendix status)](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-tiled-region-coverage-design.md)
- Upstream: [openxla/xla@5f26078f](https://github.com/openxla/xla/commit/5f26078f7cb2c6b8872c87e43fe6906923396c23) (YNNPACK reduction fusion)

---

## Appendix A: POC record

All numbers Apple M3, single thread, `bench_hlo`, seed-matched where semantics allow. Raw artifacts on the [notes branch](https://github.com/seantalts/xla/tree/notes/cpu-small-model-regression-findings).

### A.1 Dispatch micro-optimization

Per-thunk dispatch measured at 15-20 ns by a slope method (synthetic K-kernel by T-iteration while loops; the slope cancels harness floor and loop overhead). An earlier 0.5-1.2 us/thunk estimate came from a profiler artifact: nested `ThunkExecutor::Execute` frames double-count. At ~18 ns, removing all dispatch saves ~0.4 ms of jaxley's 1.1 ms, so a faster executor cannot reach the 0.36 ms result. Not pursued.

### A.2 Tiled region emission and the four-phase coverage plan

Built behind `xla_cpu_experimental_region_compilation` ([`3a1b7002`](https://github.com/seantalts/xla/commit/3a1b70020e)): routes hoisted regions through the xtile tiled emitter with fusion-view flattening (`Defuse()` to fixpoint) and non-scalar constant lifting. A dense dot/reduce region compiles to one tiled kernel, bitwise, ~35% faster on a synthetic shape. On the real workloads it was neutral: jaxley's regions are tuple-rooted and gather/reverse/DUS-heavy, outside tiled coverage. A four-phase coverage plan (multi-output, reverse, DUS, gather) was written, then shelved when A.3 showed the premise was wrong at these sizes. Findings that carry into M3: `SymbolicTileAnalysis` supports multi-output only where one root consumes the others; reverse propagates as a negative stride that emission rejects; in-place DUS risks an O(n) self-copy per iteration that parity tests cannot catch (latency must be measured).

### A.3 Merging fusions inside regions

Hypothesis: region kernels lose to the old runtime because intermediates round-trip through buffer slices inside the kernel. The IR audit supported it superficially (about 90 of 110 loads in the hottest kernel re-read just-stored values; scalar reverse and DUS loops; zero vector ops in the third-hottest kernel). The test: hand-edit the optimized HLO to merge fusion chains inside region computations so intermediates rematerialize in registers, preserving semantics. Result: the duplication-free merge moved end-to-end time about 1%, within noise, on both 3-region and 29-region variants; a variant that duplicated shared chains ran 24% slower (1.366 ms) from recomputing gather/reverse chains. All variants bitwise-identical. At 32-element shapes, in-kernel data movement is nearly free; the cost is per-kernel overhead. This measurement redirected the whole effort from codegen quality to granularity.

### A.4 In-kernel scatter emission

Tiled route: scatter has no tile propagation rule and cannot have one (data-dependent write indices); confirmed by probe. Legacy route: `HandleScatter` is unimplemented, and writing it would reproduce what `ScatterExpander` already generates, with more risk. Expansion reached the old-runtime target, so in-kernel scatter emission is deferred to M3, where it would close the remaining 0.352 vs 0.238 ms gap.

### A.5 Scatter placeholder ceiling

Before building P2, we replaced all scatters with shape-correct placeholders (semantics intentionally broken, inputs kept alive) and recompiled. The region pass folded the entire entry, main loop included, into one kernel at default gates: 0.238 ms/step. This bounded the achievable win and confirmed scatter was the only boundary that mattered. A related null result: raising the byte threshold alone changed nothing; the gate was never the limiter.

### A.6 Diagnostic misses

Recorded so they are not repeated: (a) the "frag" microbenchmark shows no win from thunk collapse because its 59 thunks dispatch once, not per loop iteration; (b) mock_mass_matrix never folds (f64 working set exceeds the byte gate) and is DUS-bound besides; (c) an apparent scalar-tanh regression was an artifact of an early prototype routing through the legacy emitter and does not exist on the shipping path; (d) a 13x dispatch estimate came from double-counted profiler frames. Each was caught by re-measuring the specific claim.

### A.7 Loop re-rolling spike (2026-07-08)

Synthetic matrix, identical math verified bitwise across forms (loop-carried stepping body: slice, 4x4 dot, tanh, dynamic-update-slice; N in {8, 32, 128, 512}; "upstream" is the sentinel configuration, "branch" is this proposal):

| form | compile | runtime |
|---|---|---|
| rolled, branch | ~13 ms, flat in N | 27 us at N=512 (~25 ns/iter, single folded kernel) |
| rolled, upstream | ~17 ms | 47 us at N=512 (~70-90 ns/iter) |
| unrolled, upstream | 69 ms (N=32), 193 ms (128), 0.8-1.1 s (512) | 24 / 34 / 152 us |
| unrolled, branch | 205 ms (N=32), 2.17 s (128); the gate stops folding at 512 (force-fold: 29.7 s) | 1.3-1.5x faster than upstream unrolled where folded |

JAX-level A/B (same computation as a Python loop vs `lax.scan`, N=128, AOT-timed): jax 0.4.30 gives unrolled 534 ms compile / 2.6 us run and scan 15 ms / 2.7 us; jax 0.10.1 gives unrolled 225 ms / 22.2 us and scan 18 ms / 4.9 us. Unrolled regressed 8.5x across versions while scan barely moved. On this branch the scan variant's HLO compiles in 21 ms and folds to a single kernel.

Compile-blowup attribution (the P3 motivation), by flag ablation on the hoisted 128-op unrolled chain: parallel-codegen split count, no effect (2184 ms either way); disable-expensive-passes, no effect; opt level 1, 1546 ms; opt level 0, 73 ms with runtime 30 us vs 27 us at O3. The cost is LLVM middle-end optimization of one giant basic block.

Why libraries unroll, from source: MJX's per-joint loops dispatch on static joint type with different widths per branch (`mujoco/mjx/_src/scan.py`), which `lax.scan` cannot express; there is no unroll option to flip. Homogeneous loops in the same libraries already roll (MJX solvers, jaxley's checkpointed scan, diffrax's while loops).

## Appendix B: Implementation guide

### B.1 Code map

| commit | files | contents |
|---|---|---|
| [`c710a3a3`](https://github.com/seantalts/xla/commit/c710a3a39d) | `xla/service/cpu/small_region_hoisting_pass.{h,cc,_test.cc}`, wiring in `cpu_compiler.cc` (~line 1152) | P1: partition entry and control-flow bodies, outline runs into `xla_cpu_small_call` calls |
| [`3a1b7002`](https://github.com/seantalts/xla/commit/3a1b70020e) | `xla/backends/cpu/codegen/emitters/region_kernel_emitter.{h,cc}`, `tiled_fusion_emitter.cc`, `thunk_emitter.cc` routing | tiled region emission, flag-gated (M3 seed) |
| [`aaa4c553`](https://github.com/seantalts/xla/commit/aaa4c553a0) | `xla/service/cpu/small_scatter_expander.{h,cc,_test.cc}`, wiring in `cpu_compiler.cc` (~line 931) | P2: byte-predicated `ScatterExpander` subclass |
| (todo) | `small_region_hoisting_pass.cc` | P3: straight-line member cap |

### B.2 Pass mechanics

`SmallRegionHoistingPass(byte_threshold, min_region_size=4, exclude_nonscalar_constants=false)`:
- Walks each computation in topological order, splitting at boundary instructions (`InstructionIsUnavailable`: custom-call, infeed/outfeed, scatter, sort, fft, partition/replica-id, custom fusions, collectives, recursively through called computations). M2 removes custom-call from this list when the runtime-context ABI lands.
- For each maximal run, liveness picks parameters (used from outside) and results (used outside or root); multi-output runs get a tuple root. Runs whose members have control dependencies crossing the boundary are skipped.
- Cost gate: summed `bytes_accessed` under the threshold, and member count >= 4 or a control-flow member. P3 adds: no-control-flow runs also need member count <= cap.
- Control-flow bodies are enqueued after their parent, so a swallowed loop body is not re-partitioned.
- `thunk_emitter.cc:663` routes tagged calls to `ComputationKernelEmitter`, or `RegionKernelEmitter` under the experimental flag with fallback.

`SmallScatterExpander(small_buffer_access_size)`:
- `InstructionMatchesPattern` = base match (all scatters) and static byte footprint (operands plus result tuple leaves) under the threshold; dynamic shapes decline.
- Pipeline position: the scatter-expansion sandwich in `cpu_compiler.cc`, fusion-emitter branch, immediately before `post_scatter_expansion_simplification`.

### B.3 Flags

- `xla_cpu_small_while_loop_byte_threshold` (backend extra option): shared threshold; unset uses defaults (region gate 64KB); 0 disables P1 and P2 (every baseline in this doc).
- `xla_cpu_experimental_region_compilation`: tiled region emission (M3 seed, off by default).
- Workaround available today for jax#26145-class models: `XLA_FLAGS=--xla_cpu_use_fusion_emitters=false` forces unconditional scatter expansion; not recommended for large models.

### B.4 Reproducing measurements

`bench_hlo` (dev tool, `xla/tools/bench_hlo.cc`, deliberately not committed) compiles an HLO file once and reports per-iteration wall time; `--profile` prints per-thunk attribution; `--print_result --seed=42` enables deterministic cross-run diffs.

```
# jaxley, branch vs upstream-equivalent
bench_hlo --hlo_file=jaxley-dump/module_0272.jit_run.before_optimizations.txt \
  --iters=100 --warmup=10 --print_result --seed=42
XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0" \
  bench_hlo ... (same)
# structure: add --xla_dump_to and count entry instructions, " scatter(", " while(" in *after_optimizations*.txt
```

### B.5 Upstream PR slicing

1. P1 + P3, replacing `SmallWhileLoopHoistingPass` (port its tests, keep its option), with the compile-time matrix as a regression test.
2. P2 on top (~60 lines plus tests), with the jaxley numbers.
3. M2 and M3 as separate reviews.

### B.6 Work checklist

- [ ] P3 cap; re-run matrix and jaxley
- [x] Large-model verification, first pass: Gemma3 1B (`gemma3_1b_flax_call`) shows the passes firing on per-layer bookkeeping with bit-identical output, runtime wash, compile +13% ([record](https://github.com/seantalts/xla/blob/notes/cpu-small-model-regression-findings/gemma3-1b-large-model-verification.md)); extend to gemma2_2b_keras_jax and gemma4_2b_bf16
- [ ] jax#37465: reshape+reduce rewrite for size == stride reduce-window, targeting XLA's own reduce emitters
- [ ] jax#33666: M2 design review (context ABI, FFI frames, error paths), then A/B on the repro
- [ ] jax#26021: DUS in-place diagnosis to confirmation or refutation, then M3 DUS work item; file the MJX batching issue with spike numbers
- [ ] M3 design review (region driver, op coverage order, compile-time tiering)
