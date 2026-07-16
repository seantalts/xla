# Modernizing XLA:CPU kernel emission: large, cost-modeled kernels without the legacy emitter

**Author:** seantalts · **Date:** 2026-07-16 · **Status:** Draft for review (scoping)
**Prior docs:** [small-model performance](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-small-model-performance-design.md) (analysis + M1/M2/M3 sketch) · [region hoisting pass](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-small-region-hoisting-pass-design.md) (includes glossary) · [scatter expander](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-small-scatter-expander-design.md) · [gating experiment brief](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-region-gating-experiment-brief.md) · [tiled coverage findings](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-tiled-region-coverage-design.md)

## Summary

The small-model work has so far been a retrofit: region hoisting plus scatter expansion recover pre-thunk performance (jax#26145: 1.53 to 0.36 ms, bitwise), but the region backend is the legacy `IrEmitter`, the boundary list is a fixed op-capability table, libraries are unreachable from inside kernels, and fusion sizing is heuristic. This doc scopes the end state and the path to it:

- **One kernel construct.** A "large fusion with materialized intermediates" and a "region" are the same object. The end state stops distinguishing them: a kernel is a set of instructions emitted as one function, where a cost model decides per producer-consumer edge whether a value is rematerialized, kept in a tile-sized stack allocation, or materialized to a scratch buffer. Multi-output fusion becomes one point on that spectrum rather than a special mechanism.
- **No fixed blockers.** Every op today classed "unavailable" either gains a modern emission path, an HLO expansion, or a mid-kernel library/runtime call. The only permanent boundaries are ops that suspend execution (infeed/outfeed, collectives).
- **Libraries callable mid-kernel.** A runtime-context kernel ABI lets emitted code call YNNPACK, oneDNN, Eigen, LAPACK/FFI, and host callbacks from inside a kernel, the way the pre-thunk emitter called Eigen inline for years. Per project direction, the cost model defaults to native codegen and reaches for a library only where it is clearly better (large dots/convs); this work makes placement possible, not more library-hungry.
- **xtile as the only backend.** The region driver, op-coverage completion, and intra-kernel parallelism land on the xtile/MLIR path, after which `IrEmitter`, `IrEmitter2`, and `ComputationKernelEmitter` are deleted.
- **Decisions by cost model.** The static, machine-parameterized model (dispatch and ABI savings amplified by trip count, recompute, overlap loss, register pressure, stream count, working set, library deltas, code size) replaces the byte gates, the amplification boolean, the min-size counter, fusion-pass duplication caps, and module gates. Every term is order-of-magnitude computable from the graph plus a machine description; the model's job is to avoid the catastrophic corners we have cataloged, all of which it catches.

## Where emission stands today

| path | used for | status |
|---|---|---|
| xtile / fusion emitters (MLIR) | standalone loop fusions, scatter fusions, concat/transpose/etc. | modern; the base to build on |
| tiled emitter (`EmitTiledComputation`) | experimental; dense-math regions behind `xla_cpu_experimental_region_compilation` | modern; limited op coverage, single-array roots |
| `ComputationKernelEmitter` -> legacy `IrEmitter` | region kernels (`xla_cpu_small_call`), control flow inside regions | legacy; runtime calls disabled ([`computation_kernel_emitter.cc:248`](https://github.com/seantalts/xla/blob/feat/cpu-small-region-hoisting/xla/backends/cpu/codegen/computation_kernel_emitter.cc)) |
| legacy `IrEmitter`/`IrEmitter2` elemental paths | reduce-window (scalar, the jax#37465 cost), assorted fallbacks, everything under `--xla_cpu_use_fusion_emitters=false` | legacy; retirement target |
| library thunks (YNN, oneDNN, Eigen dot, FFI custom calls) | dots, convs, claimed reductions, LAPACK, callbacks | fine as thunks; unreachable from inside kernels |
| thunk-level control flow (`WhileThunk`, `ConditionalThunk`) | all loops not folded into regions | fine; regions subsume the hot small ones |

## Blocker inventory and resolutions

Every reason an instruction cannot join a kernel today, and how it goes away:

| blocker | today | resolution | phase |
|---|---|---|---|
| scatter | expanded when small (shipped); boundary when large | keep expansion; optional in-kernel sequential emission closes the last measured gap (0.352 vs 0.238 ms) | shipped / P3 |
| custom-call / FFI (LAPACK, callbacks) | boundary; blocks jax#33666 | runtime-context ABI + prebuilt call frames; call mid-kernel | P1 |
| sort | boundary (runtime KeyValueSort) | library call-out through the same ABI | P1 |
| fft | boundary (runtime DUCC call) | library call-out | P1 |
| copies | fold today, losing the memcpy fast path in kernels | emit `llvm.memcpy` for in-kernel copies; classification becomes unnecessary | P1 (trivial) |
| loop fusions protected for new emitters | amplification-gated availability (internal) | moot once regions emit through the same xtile emitters as standalone fusions | P2 |
| tuple-rooted (multi-output) regions | tiled path declines; legacy handles | multi-output support in `SymbolicTileAnalysis` (relax the "real root" form for independent roots under whole-shape tiles) and the region driver | P2 |
| reduce-window | scalar legacy emission (the jax#37465 12 us) | tiled reduce-window (tiling space + propagation + emitter case), or the reshape+reduce rewrite as the interim | P3 |
| gather / reverse / dynamic-update-slice | tiled path lacks cases | emitter cases per the [coverage findings](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-tiled-region-coverage-design.md): gather as per-row dynamic-slice, reverse as negative-stride normalization, DUS with in-place discipline and a latency (not parity) gate | P3 |
| control flow (while/conditional) inside kernels | legacy emitter only | `scf.while`/`scf.if` region driver on xtile | P2 |
| big tensors (byte gates) | excluded to preserve parallelism | intra-kernel parallelism via the existing workgroup ABI (tile-to-thread mapping) plus cost model; gates become model outputs, not constants | P4 |
| infeed/outfeed, collectives | boundary | stays a boundary (execution suspension is a runtime property) | never |

## New emitters and emitter capabilities (the build list)

1. **Region driver on xtile (the big one, P2).** A schedule-walking emitter that takes a region computation (the `xla_cpu_small_call` callee) and emits one MLIR function: members emitted in order through the existing xtile per-op emitters, intermediates as tile allocas or scratch-buffer stores, `scf.while`/`scf.if` for control flow with loop-carried values as memrefs, tuple roots mapped to multiple result buffers. Prior work to reuse: `EmitTiledComputation` (schedule walk, per-root tiles), the flag-gated region emitter's fusion-view flattening and constant lifting, `PromoteBuffersToStack` for allocas. Parity bar: bitwise on jaxley, 24501, and Gemma3 region shapes, at equal or better latency and compile time than `ComputationKernelEmitter`.
2. **Multi-output tiling (P2).** `SymbolicTileAnalysis` accepts independent roots under the region path's whole-shape single-tile strategy (no cross-root tile-consistency problem exists there); the driver already returns per-root values.
3. **Tiled reduce-window (P3).** Window dims in the tiling space, the windowed affine map in propagation, a masked `stablehlo.reduce` emission case; vectorization then comes free from the existing reduce lowering. Closes jax#37465 permanently (the interim HLO rewrite for size == stride windows can ship earlier and independently).
4. **Tiled gather, reverse, dynamic-update-slice (P3).** As inventoried above; scope gather to observed forms; gate DUS on latency because the in-place O(n) self-copy trap passes parity checks.
5. **In-kernel sequential scatter (P3, optional).** A non-tiled loop-over-index-rows emission under whole-shape tiles; only if a workload demands the last 30% on scatter-heavy loops.
6. **In-kernel library call emission (P1).** Not an emitter for an op class but a capability all emitters share: emit a call to a registered trampoline with operands materialized to buffers, the runtime context threaded as a kernel argument, and results as ordinary buffers. Covers FFI custom calls, sort, fft, Eigen/oneDNN/YNN entry points, and host callbacks.
7. **Parallel tiles (P4).** Emit tile loops mapped onto the existing kernel workgroup dimensions so one large kernel uses the intra-op pool, replacing `ParallelTaskAssigner`'s per-op partitioning for fused/region kernels. This is what lets the byte gates relax without losing parallelism.
8. **Memcpy-aware copy emission (P1).** In-kernel copies lower to `llvm.memcpy` instead of loops.

## The runtime-context kernel ABI (P1)

Extend the host kernel calling convention with one pointer to a per-execution context (run options, intra-op threadpool, allocator, FFI execution context, profiler hooks). The thunk owns it at execute time and passes it through. Library and FFI call sites are resolved at emission time into prebuilt call frames stored with the executable; the emitted call loads the frame, fills operand pointers, and invokes a stable C trampoline. Discipline: call-outs from sequential kernel context by default; a library that manages its own parallelism receives the pool from the context rather than nesting; error status propagates through the existing kernel error return. The pre-thunk emitter's `allow_runtime_calls` path is the feasibility proof; this is its thunk-era replacement, designed once for all libraries rather than per-library.

Measurement gate: the diffrax stiff repro (jax#33666) folds its Newton loop with LAPACK called in-kernel and recovers its regression; host-callback progress bars (the 24501 with-callback variant) stop forcing the loop apart.

## The cost model (P4, scaffolded from P0)

One function, three clients (region former, fusion sizing, library placement), replacing today's constants:

- benefit: `saved_thunk_executions x ~50ns x amplification` (trip counts from `known_trip_count`, boolean fallback), plus locality gains for materialized-to-tile edges
- costs: recompute (per shared producer: internal uses x producer cost, exact from the graph), inter-op overlap loss (subgraph work minus critical path, capped by pool width), intra-op parallelism forfeited (would members partition standalone), register pressure (max live cut of the per-element expression vs physical registers; decides rematerialize vs tile-materialize per edge), operand streams vs prefetcher budget, working set vs L1/L2 (decides tile vs scratch), library delta (calibrated table; per project direction the tie goes to native codegen), code size, and a compile-time budget (member and basic-block caps as hard rails)
- machine description: dispatch constant, ABI constant, register file, cache sizes, stream budget, pool width; calibrated by the slope-method microbenchmarks and pinned by the regression anchors (jaxley, 24501, inference_gym, diffrax, jax.b380442861, the scan-with-protected-fusions case, Gemma3)

Staging: P0 keeps the shipped discrete gates (byte budget, amplification, thunk-generating min size, member cap) and adds fold-classes; P4 swaps them for the quantitative model once the anchors give it a validation surface. The gating experiment brief's corpus is the calibration set; production fold telemetry is the drift alarm.

## Phases

- **P0 (now, mostly shipped):** region hoisting + segmentation + amplification, scatter expansion, straight-line member cap, fold-classes, fold telemetry, calibration harness. Small-model issue solved on the legacy backend; everything after this improves how, not whether.
- **P1: runtime-context ABI and call-outs.** New kernel ABI parameter, trampoline registry, FFI frames, memcpy copies. Deliverables: diffrax folds (jax#33666 closed), sort/fft leave the boundary list, callbacks stop splitting loops. Independent of xtile work.
- **P2: xtile region driver.** Control flow, multi-output, buffer materialization; regions switch backends behind the existing experimental flag, then by default at the parity bar. `ComputationKernelEmitter` becomes dead code.
- **P3: op coverage.** reduce-window (jax#37465 closed), gather/reverse/DUS (jax#26021 DUS work item), optional in-kernel scatter. Each op lands with a bug-shaped test and a latency gate.
- **P4: cost-modeled boundaries.** Quantitative model replaces gates; fusion sizing and region formation unify (larger fusions with per-edge materialization); parallel tiles let big kernels keep the pool; module gates and protection lists are deleted.
- **P5: legacy deletion.** Inventory every remaining route into `IrEmitter`/`IrEmitter2` (reduce-window fallback dies in P3, region backend in P2, `use_fusion_emitters=false` mode retired with a deprecation window), delete the emitters and the `ComputationKernelEmitter`, simplify the scatter-expansion sandwich and the pipeline flags.

Ordering rationale: P1 and P2 are independent and can run in parallel; P3 needs P2's driver for the region cases but the reduce-window interim rewrite does not; P4 needs P1+P2 (the model's choices must all be emittable before it chooses them); P5 is bookkeeping once P2/P3 land.

## Issue map

| issue | closed by |
|---|---|
| jax#26145 (jaxley) | P0 (shipped, 0.36 ms bitwise) |
| jax discussion 24501 | P0 (shipped, 6-18x); callback variant improves further at P1 |
| jax#33666 (diffrax) | P1 |
| jax#37465 (reduce-window) | interim rewrite anytime; permanently P3 |
| jax#26021 (MJX) | P3 (DUS) + P4 (fusion sizing) + upstream batching per the M4 notes |
| TORAX compile time | out of scope here (inliner prevention workstream); P4's compile budget helps at the margin |

## Risks

- **P2 parity risk:** the legacy emitter is battle-tested on control flow; the xtile driver replays that maturation. Mitigation: bitwise parity harness already exists and every region shape in the corpus is a test; keep the legacy fallback flag until a full release cycle passes clean.
- **P1 reentrancy and pool discipline:** library calls from kernels can deadlock or oversubscribe if context rules are loose. Mitigation: sequential-context default, explicit pool handoff, the diffrax anchor plus a stress test with nested scans.
- **Cost model overfit to anchors:** mitigated by keeping hard rails (compile budget, register cap) non-negotiable and by the partial-model-slice rule from the brief: per-run properties only.
- **Scope creep in P3:** op coverage is issue-driven with per-op gates; the four-phase-plan history is the cautionary tale, and its findings are the reusable inputs.
- **Parallel tiles (P4) interact with the executor's own parallelism:** decide per kernel, never both; the cost model owns the choice.

## Out of scope

Loop re-rolling and inliner prevention (M4 of the parent doc); GPU-shared tiling productionization beyond what multi-output needs; YNNPACK coverage expansion (project direction is less to the library, not more); collectives/infeed inside kernels.
