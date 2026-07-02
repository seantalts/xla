# Tiled-Emitter Coverage for Region Compilation: Multi-Output, Reverse, Dynamic-Update-Slice, Gather

**Status:** Design for review · **Date:** 2026-07-02
**Companion:** "Single-Kernel Region Emission for XLA:CPU" (branch `design/cpu-stage1-region-compilation`), which established emitting a hoisted region as one tiled MLIR kernel and shipped the straight-line dense-math milestone (dot/reduce/broadcast/elementwise regions fold, flag-gated).

## Summary

Region compilation folds a maximal run of adjacent instructions into a single tiled kernel, keeping intermediates in registers/stack instead of round-tripping buffer slices. The shipped milestone covers single-array-root regions of dense-math ops. Profiling the flagship scatter-heavy workload (a branched-cable neuron simulation, JAX #26145) showed its 25 hoisted loop-body regions decline for a *stack* of reasons: every region is **tuple-rooted** (multi-output), and their bodies are pervaded by three ops the tiled emitter cannot emit — **gather** (~91), **reverse** (~72), and **dynamic-update-slice** (~56). (Scatter itself is un-tileable — data-dependent write indices admit no static tile relation — and correctly stays a region *boundary*; nothing here attempts scatter.)

This doc plans the four missing emitter features as **risk-ordered phases with a measurement gate after each**. The honest framing up front: for #26145 specifically, region *folding* is already delivered by the legacy region emitter (~1.33× committed); the tiled path's marginal value there is codegen quality (vectorization, stack-localized intermediates) on already-folded regions plus folding under the tiled flag. Coverage is **multiplicative** — a region folds only if *every* member op is supported — so partial coverage may fold few regions. That is why each phase ends with a fold-rate/latency measurement and an explicit continue/stop decision, rather than committing to all four features blind.

## Current state (verified at head of `feat/cpu-small-region-hoisting`)

| capability | tiling analysis | tiled emission (`EmitTiledHloInstruction`) |
|---|---|---|
| multi-output (tuple root) | **partial** — supported only when one "real root" transitively consumes all other roots (`symbolic_tile_analysis.h:48-67`) | driver already returns one tile value per root (`fusion_emitter.cc:1034,1063-1065` iterates `tiled_computation.roots()`) |
| reverse | mentioned once in `symbolic_tile_analysis.cc` (verify scope) | **no case** |
| dynamic-update-slice | **no rule** | **no case** (but `dynamic-slice` — the read-side dual — is supported) |
| gather | **no rule** | **no case** |

Region-side gates that decline today: `region_kernel_emitter.cc` rejects non-array roots and `kTuple`/`kGetTupleElement`; the CPU allow-list (`tiled_fusion_emitter.cc:177 IsSupportedInstruction`) rejects all three ops.

## Phase 0 — measurement harness (prerequisite, ~1 day)

Add fold-rate telemetry: when the region flag is on, log/count per module how many hoisted regions folded via the tiled path vs declined, and the first decline reason (tuple root / specific unsupported op / tiling failure). Every later phase's gate reads this. Also pin the #26145 benchmark numbers (flag off / legacy folding) as the baseline.

## Phase 1 — multi-output (tuple-root) regions [the gate for everything else]

**What:** emit a region whose root is a tuple of N independently-live values as one tiled kernel with N result buffers.

**Design sketch:**
- Region emitter: lift the `!root->shape().IsArray()` and `kTuple`/`kGetTupleElement` rejections; build a tuple-root fusion view; map each tuple element to a `KernelSpec` result buffer (the kernel ABI already supports multiple results; buffer slices come from the call's tuple shape).
- Tiling: the driver emits per-root tiles already. The open question is the analysis: `SymbolicTileAnalysis` currently accepts multi-output only in "real root consumes the other roots" form. The target regions have **independent** roots (several live values feeding different downstream scatters). **First task of this phase is a probe:** run the analysis on a real tuple-root region view and capture whether independent roots are rejected. If rejected, two options, in preference order: (a) relax the analysis for the CPU/region case where all roots share the single-tile ("whole shape") tiling the region path uses — with one tile per root there is no cross-root tile-consistency problem to solve; (b) split the region into per-root sub-fusions sharing the flattened body (loses cross-root CSE — a fallback, not the plan).
- Multi-root tile selection: `GetTiling` currently derives tile sizes from one root; under the region path's single-tile strategy each root's tile is its whole shape, so selection is per-root and trivial. Document this as region-path-only; general multi-root tile policy stays out of scope.

**Risks:** the analysis relaxation touches shared (GPU-shared) tiling code — keep it behind the region path; result aliasing (regions pass scatter buffers through — output buffers must respect existing aliasing).

**Gate (measure before Phase 2):** fold-rate on #26145 with multi-output alone. Expectation: still ~0 regions fold (bodies contain gather/reverse/DUS), but the decline reason shifts from "tuple root" to specific ops — confirming the harness and sizing each op's blocking share. Plus: a synthetic multi-output dense region folds, bitwise parity, no regression on the shipped single-root path. If the analysis relaxation proves deep, stop and reassess here.

## Phase 2 — reverse (cheapest op, ~days)

**What:** `reverse(x, dims)` inside a region. Indexing is a static affine mirror (`i → size-1-i` on reversed dims) — the easiest possible tile rule.

**Design sketch:** add reverse tile propagation (an affine map negating the reversed dims' strides/offsets — same family as `slice`, which is supported) and an `EmitTiledHloInstruction` case (extract the mirrored input tile, `stablehlo.reverse` or reversed-stride extract on the tile). Verify what the existing lone `Reverse` mention in `symbolic_tile_analysis.cc` covers — it may be partially present.

**Gate:** fold-rate delta on #26145; synthetic reverse-containing region bitwise parity.

## Phase 3 — dynamic-update-slice (~1 week)

**What:** `dynamic-update-slice(operand, update, indices…)` inside a region — the write-side dual of the already-supported `dynamic-slice`.

**Design sketch:** tiling — output shape equals operand shape; the update occupies a runtime-offset window. Under the region path's single-tile strategy the tile is the whole array, so the rule degenerates to "operand tile = output tile; update tile = whole update at runtime offset" — no general strided-window tiling needed for v1. Emission — copy/forward the operand tile, then insert the update tile at the runtime offset (`tensor.insert_slice` with dynamic offsets), mirroring how the `dynamic-slice` case reads at a runtime offset. In-place semantics: XLA's DUS-in-place optimization (operand and result share a buffer) must be preserved or safely bypassed — under the region path intermediates are region-local SSA, and only region-escaping DUS results have buffers; verify the aliasing story on a real region before implementing.

**Risks:** in-place/aliasing is the sharp edge; out-of-bounds clamping semantics of DUS offsets must match HLO (clamp, not wrap).

**Gate:** fold-rate delta on #26145 (with reverse + DUS + multi-output, a meaningful subset of regions may now fold); bitwise parity on DUS-containing synthetic regions; explicitly test the DUS-result-escapes-region aliasing case.

## Phase 4 — gather (hardest, ~1–2 weeks)

**What:** `gather(operand, indices)` inside a region. Unlike scatter, gather is *tileable in principle*: the output→indices mapping is affine; only the operand access is runtime-indexed — per output element it is a dynamic-slice of the operand driven by the indices value.

**Design sketch:** tiling — the output tile maps affinely to an indices tile; the operand is accessed like `dynamic-slice` (whole-operand availability under the single-tile strategy, so no operand tiling needed for v1). Emission — an `EmitTiledHloInstruction` case that loads the indices tile, then per output row extracts the operand window at the runtime index (loop or `tensor.extract`-based; vectorize the trailing window dim). This is the most new code and the least precedent; scope v1 to the simple gather forms the workload actually uses (verify against the dumped gathers: `update_window_dims`, `index_vector_dim` shapes) rather than general gather.

**Risks:** general gather is a large semantic surface — scope to observed forms; performance of row-at-a-time gather inside a tile may need iteration.

**Gate (the payoff measurement):** #26145 fold-rate and end-to-end latency vs both baselines (legacy folding, flag off). This is where the tiled path either beats the legacy region emitter's ~1.33× materially (vectorized bodies + stack intermediates) or doesn't — the final continue/stop decision for this whole line.

## Sequencing rationale

Multi-output first because nothing folds without it and it gates all measurement; then strictly by risk/effort (reverse trivial-affine → DUS with a supported dual → gather greenfield). Coverage being multiplicative means intermediate gates mostly measure *decline-reason shifts* rather than wins; the honest expectation is that the latency payoff, if any, arrives only at Phase 4. The per-phase gates exist precisely so the investment can stop early if the fold-rate math or the Phase-1 analysis probe says the payoff won't materialize — for #26145 the folding win is already banked on the legacy path, so this line must justify itself on codegen quality.

## Out of scope

Scatter emission inside a tile (un-tileable, stays a boundary); control flow inside regions (separate milestone); general multi-root tile-size policy beyond the region path's single-tile strategy; productionizing tiling propagation (separate parallel work the region path composes).

## Key code references

`xla/codegen/tiling/symbolic_tile_analysis.{h:48-67,cc}` (multi-output "real root" constraint) · `xla/codegen/xtile/codegen/fusion_emitter.cc:887-1032` (`EmitTiledHloInstruction` op cases; dynamic-slice precedent), `:1034-1065` (multi-root driver) · `xla/codegen/tiling/tile_propagation.cc` (per-op tile rules; slice as the reverse/DUS template) · `xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.cc:177` (CPU allow-list), `:159` (`GetTiling`) · `xla/backends/cpu/codegen/emitters/region_kernel_emitter.cc` (region gates to lift) · #26145 region bodies: `perf-regression/jaxley-dump/module_0272.jit_run.before_optimizations.txt`.
