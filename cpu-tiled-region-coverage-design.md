# Region Compilation Coverage for XLA:CPU — Measurement Update and Revised Plan

**Status:** Revised after Phase-0.5 measurements · **Date:** 2026-07-02 (originally published earlier the same day as a four-phase tiled-emitter coverage plan; that plan is preserved as an appendix)
**Companion docs:** "Single-Kernel Region Emission for XLA:CPU" (same branch) — the region-hoisting + region-emission design this builds on; `cpu-phase05-dispatch-decomposition-findings.md` (branch `notes/cpu-small-model-regression-findings`) — the full measurement record behind this revision.

## Summary

Region compilation folds a maximal run of adjacent instructions into a single kernel, eliminating per-thunk dispatch and ABI overhead. The original version of this doc planned four tiled-emitter features (multi-output, reverse, dynamic-update-slice, gather) to extend tiled-MLIR coverage to the flagship scatter-heavy workload (jaxley branched-cable neuron simulation, JAX #26145). Before building, we profiled and ran hand-edited-HLO POCs to verify the target — and the measurements redirect the plan entirely:

1. **The workload is thunk-granularity-bound, not codegen-bound.** ~22.4k thunk executions per step at ~50 ns each (≈18 ns dispatch + ≈30 ns kernel ABI/compute on ≤32-element arrays) account for the residual 3.0× gap to the old runtime. Hand-merging fusions inside region kernels (the win a better in-kernel emitter would deliver) moved end-to-end latency ~1% — noise.
2. **Scatter boundaries are the whole game.** With scatters out of the way, the existing hoisting pass swallows the *entire program* — the 300-iteration while loop included — into ONE kernel, at default flags (control-flow regions already bypass the min-size gate).
3. **The fix already exists in-tree.** `ScatterExpander(kEliminateAllScatters)` (`cpu_compiler.cc:927-929`) lowers scatter to while+gather+dynamic-update-slice — all ops the legacy region emitter emits natively. Forcing it via the existing `--xla_cpu_use_fusion_emitters=false` flag: **1.09 → 0.352 ms/step, bitwise-identical results, beating the old-runtime target of 0.372 ms** — full recovery of the #26145 regression with zero new code.

The revised plan is therefore a small, targeted feature — **selective scatter expansion** — instead of 4–6 weeks of tiled-emitter work. The four-phase tiled plan is retained below as an appendix, no longer justified by #26145.

## Measured results (Phase 0.5)

Apple M3, single thread, `bench_hlo`, 100 timed iterations; workload = jaxley `module_0272` (JAX #26145).

| configuration | ms/step | numerics |
|---|---|---|
| thunk runtime, hoisting off | 1.531 | baseline |
| + region hoisting (shipped Stage-0: 18 regions + 31 scatters + ~24 other thunks per while iteration) | 1.09–1.12 | bitwise |
| POC-1: hand-merged fusions *inside* region kernels | ~1.10 (no change) | bitwise |
| POC-2: scatters → passthroughs, whole entry = 1 kernel (ceiling probe) | 0.238 | broken by design |
| **POC-3: scatters expanded via existing pipeline (flag only)** | **0.352** | **bitwise (seed-matched)** |
| old runtime (jax 0.4.30, end-to-end) | 0.372 | — |

Supporting evidence (full record in the findings doc and its `phase05-artifacts/`):
- **Thunk census:** 300 while iterations/step × ~73 body thunks + ~144 entry thunks ≈ 22.4k thunk executions/step. Hoisted region kCalls are single kernel thunks (`thunk_emitter.cc:663-700`), confirmed by executor-invocation counts (301/step).
- **IR audit of hot region kernels:** genuinely imperfect code (≈90 unforwarded store/reload round-trips in `call.97_kernel` due to a `!noalias` scope gap around memcpys; scalar reverse/DUS loops; ~28 arg loads + 19 tuple stores ABI per call) — yet POC-1 proves fixing the *bodies* is worth ~nothing at these shapes. The cost is around kernels, not inside them.
- **Why scatter blocks regions:** legacy `IrEmitter::HandleScatter` is `Unimplemented` (`ir_emitter.cc:1896`), so the hoisting pass treats scatter (and any fusion containing one) as a region boundary.
- **POC-3 structure:** optimized module has entry = 1 call; 0 scatters; 55 whiles (1 main + 54 expanded scatters) as native LLVM control flow inside one function. The 0.352-vs-0.238 delta is the real scatter compute POC-2 faked away.

## Revised plan: selective scatter expansion

**What:** expand scatters whose operand+updates+indices byte volume is small — aligned with the region-hoisting byte gate (64 KB default) — into while+gather+DUS, at the pipeline position the unconditional expander already occupies (`cpu_compiler.cc:921-929`, sandwiched before post-expansion simplification, fusion, and copy insertion, so all downstream machinery processes the expanded form normally). Large scatters keep the dedicated MLIR scatter-fusion kernel. Everything downstream — hoisting, control-flow-in-region, legacy single-kernel emission — already works and is already committed.

**Why this position:** expansion must precede copy insertion (the expanded while loops need loop-carried copy semantics) and benefits from preceding simplification+fusion (expanded components get simplified and fused, which also softens the cost if a given scatter ends up *not* hoisted).

**Key risk — mispredicted expansion:** an expanded scatter that hoisting does *not* later swallow becomes a while *thunk* dispatching per-index-row body thunks — much slower than the dedicated scatter kernel. Mitigations, in order: (a) conservative size predicate; (b) an explicit regression benchmark with large/non-hoistable scatters that must not match the predicate; (c) if v1 mispredicts in practice, a v2 predicate that dry-runs the hoisting pass's region-eligibility analysis ("would this scatter fragment an otherwise-hoistable region?") — the analysis is already factored for reuse in `small_region_hoisting_pass.cc`.

**Verification sequence (each step gated):**
1. Reproduce the POC-3 collapse with fusion emitters ON and expansion forced for all scatters (POC-3 had them off globally; hoisting is HLO-level and region internals use legacy emission regardless, so the result should carry — measure it).
2. Land the selective pass behind the predicate; jaxley gate: bitwise parity, ≤0.4 ms/step, whole-entry (or near) collapse in the after-opt dump.
3. Non-regression gate: large-scatter workload latency unchanged; full hoisting/regression test suites green.
4. Bank and re-measure the three JAX issues; #26145 should close.

**User-visible workaround available today:** `XLA_FLAGS=--xla_cpu_use_fusion_emitters=false` recovers the full regression for small scatter-heavy models right now, at the cost of disabling MLIR fusion emitters globally (unwise for large models).

## Scope

This addresses #26145. #37465 (scalar reduce-window emission) and #33666 (LAPACK FFI + Python callbacks in a Newton loop) have unrelated root causes and separate workstreams. Scatter *emission inside a kernel* (tiled or legacy `HandleScatter`) is no longer needed for #26145 but would close the remaining 0.352→0.238 gap if a workload ever demands it; it stays out of scope.

---

## Appendix: original four-phase tiled-emitter coverage plan (superseded for #26145)

Preserved for reference: the plan below extends *tiled-MLIR* region emission coverage. Phase-0.5 measurements showed its premise — that in-kernel codegen quality was the residual bottleneck — does not hold on #26145 (POC-1: ~1% effect). Revisit only if a workload with larger per-kernel shapes makes tiled region emission pay; the per-op findings (multi-output "real root" constraint, reverse negative-stride gap, DUS in-place trap) remain valid and are kept current below.

### Current state (verified at head of `feat/cpu-small-region-hoisting`)

| capability | tiling analysis | tiled emission (`EmitTiledHloInstruction`) |
|---|---|---|
| multi-output (tuple root) | **partial** — supported only when one "real root" transitively consumes all other roots (`symbolic_tile_analysis.h:48-67`) | driver already returns one tile value per root (`fusion_emitter.cc:1034,1063-1065` iterates `tiled_computation.roots()`) |
| reverse | propagates as a **negative stride**, hard-rejected at emission (`symbolic_tile_analysis.cc:1671-1683`) | **no case** |
| dynamic-update-slice | **no rule** | **no case** (but `dynamic-slice` — the read-side dual — is supported) |
| gather | **no rule** | **no case** |

Region-side gates that decline today: `region_kernel_emitter.cc` rejects non-array roots and `kTuple`/`kGetTupleElement`; the CPU allow-list (`tiled_fusion_emitter.cc:177 IsSupportedInstruction`) rejects all three ops.

### Phase 1 — multi-output (tuple-root) regions

**What:** emit a region whose root is a tuple of N independently-live values as one tiled kernel with N result buffers.

**Design sketch:**
- Region emitter: lift the `!root->shape().IsArray()` and `kTuple`/`kGetTupleElement` rejections; build a tuple-root fusion view; map each tuple element to a `KernelSpec` result buffer.
- Tiling: `SymbolicTileAnalysis` accepts multi-output only in "real root consumes the other roots" form; the target regions have **independent** roots. First task is a probe on a real tuple-root region view. If rejected, prefer relaxing the analysis for the region case where all roots share the single-tile ("whole shape") tiling — with one tile per root there is no cross-root tile-consistency problem; fallback is per-root sub-fusions sharing the flattened body (loses cross-root CSE).
- Multi-root tile selection: under the region path's single-tile strategy each root's tile is its whole shape — selection is per-root and trivial; general multi-root tile policy stays out of scope.

**Risks:** the analysis relaxation touches GPU-shared tiling code — keep it behind the region path; result aliasing must respect existing buffer aliasing.

### Phase 2 — reverse

**Design sketch:** propagation already produces the mirrored map — reverse comes out of the shared symbolic-tile machinery as a *negative stride*, and `ComputeTiledComputationImpl` (`symbolic_tile_analysis.cc:1671-1683`) hard-rejects any negative stride with `UnimplementedError`. The in-code comment already suggests the fix: normalize strides to ≥ 0 and emit reversed reads. So the work is negative-stride handling in codegen, not a new tile rule. Under the region path's whole-shape single-tile strategy this is especially simple: a full tile with stride −1 is just an in-kernel reverse of the extracted tile (offset normalization is trivial at offset 0).

### Phase 3 — dynamic-update-slice

**Design sketch:** output shape equals operand shape; under the single-tile strategy the rule degenerates to "operand tile = output tile; update tile = whole update at runtime offset". Emission: forward the operand tile, `tensor.insert_slice` at the runtime (clamped) offset, mirroring the supported `dynamic-slice` read case.

**Risks:** in-place/aliasing is the sharp edge; out-of-bounds clamping must match HLO. **Performance trap:** when DUS is in-place (operand and result share a buffer — the loop-carried-state pattern), the "copy operand tile, then insert" lowering degenerates to a full O(n) array copy per loop iteration unless bufferization elides the self-copy — passes bitwise parity while regressing latency, so any gate must measure latency on an in-place loop-carried DUS region, not just parity.

### Phase 4 — gather

**Design sketch:** unlike scatter, gather is tileable in principle: the output→indices mapping is affine; only the operand access is runtime-indexed (per output element, a dynamic-slice driven by the indices value). Emission: load the indices tile, per output row extract the operand window at the runtime index; vectorize the trailing window dim. Scope v1 to the observed gather forms (`offset_dims`, `collapsed_slice_dims`, `start_index_map`, `index_vector_dim`) rather than general gather. Note: the IR audit found dynamic-index gathers cannot vectorize on NEON — temper expectations accordingly.

### Key code references

`xla/codegen/tiling/symbolic_tile_analysis.{h:48-67,cc:1671-1683}` · `xla/codegen/xtile/codegen/fusion_emitter.cc:887-1065` · `xla/codegen/tiling/tile_propagation.cc` · `xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.cc:{159,177}` · `xla/backends/cpu/codegen/emitters/region_kernel_emitter.cc` · `xla/service/cpu/{cpu_compiler.cc:921-929,ir_emitter.cc:1896,thunk_emitter.cc:663-700}` · workload: `perf-regression/jaxley-dump/module_0272.jit_run.before_optimizations.txt`.
