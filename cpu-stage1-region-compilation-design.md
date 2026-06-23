# Stage-1 Region Compilation for XLA:CPU — Design

**Author:** seantalts · **Date:** 2026-06-23 · **Status:** Design for review
**Predecessors:** `cpu-region-compilation-design.md` (the overall Stage 0/1/2 plan) · `cpu-stage1-region-emission-scoping.md` (feasibility record, all risks retired) · `cpu-small-model-regression-findings.md` (root-cause map)
**Landed:** Stage-0 (`c710a3a39d`) — body-internal region hoisting via the legacy `ComputationKernelEmitter`, jaxley #26145 1.33×.

## Abstract

The thunk runtime executes each top-level HLO instruction as its own kernel + thunk. For loop-structured small models (ODE solvers, neuron sims, MPC) this is a large regression: the per-iteration body is dozens of tiny kernels, and the cost is dominated **not** by per-thunk dispatch (~15–20 ns, measured) but by **per-kernel granularity** — every intermediate round-trips through a buffer slice instead of a register, and there is no optimization across kernel boundaries. Profiling decomposed jaxley's ~3× residual as ~74% memory-round-trips / ~26% dispatch; the old non-thunk runtime avoided both by emitting the whole loop body as one inlined, cross-optimized function.

Stage-0 recovered the part of this tax that lives *around* ops the legacy emitter can't handle (it collapsed jaxley's non-scatter runs, 1.33×). The remaining ~3× requires collapsing *across* scatter and across differing shapes into one kernel. **Stage-1 emits a region as a single kernel on the MLIR/xtile tiled-emitter stack**, which (a) already has a multi-op, multi-shape, tensor-threading driver (`EmitTiledComputation`), (b) emits scatter (`CpuScatterFusion`), and (c) localizes non-escaping intermediates to `alloca` automatically via bufferization. The region *mechanism* (HLO-level partitioning + an attribute) is unchanged from Stage-0; only the *executor* behind it changes — the migration-safe decomposition from the overall design.

## 1. Goal

Take an outlined region — a maximal schedule-contiguous run of region-eligible instructions, produced by the existing `SmallRegionHoistingPass` — and emit it as **one tiled MLIR kernel** with intermediates in registers/allocas and cross-op optimization, instead of N per-op kernels (Stage-0/thunk) or one legacy-IrEmitter kernel that cannot express scatter (Stage-0's `ComputationKernelEmitter`).

Target: close the bulk of jaxley's remaining ~3× (ceiling ≈ the old 373 µs; the scatters are tiny `f32[32]`/scalar, so the residual is granularity, not intrinsic scatter compute).

## 2. Design

### 2.1 Region → fusion → tiled emitter

The region's outlined computation is routed to the CPU **xtile tiled fusion emitter** rather than the legacy `ComputationKernelEmitter`. The tiled emitter's `EmitTiledComputation` (`xla/codegen/xtile/codegen/fusion_emitter.cc:1034`) is precisely the region driver the overall design called for: it walks the (tiled) computation in schedule order, emits each instruction via `EmitTiledHloInstruction` → a `tensor`-typed SSA value, and threads results through an instruction→`TensorValue` map. It natively handles dot, reduce, broadcast, pad, concatenate, transpose, reshape, and elementwise — i.e. the full op set of the regressing workloads — and intermediates of differing shapes compose because each is a real tensor value, not a shared per-output-element index domain.

Two mechanisms reach it; v1 picks whichever is cleaner (decided during implementation, recorded in the scoping doc):
- **(A) region-as-fusion:** `SmallRegionHoistingPass`, under the Stage-1 flag, emits the region as a `kFusion` (kLoop) whose fused computation is the region, instead of a `kCall` tagged `xla_cpu_small_call`. The existing fusion/tiled path then applies unchanged.
- **(B) fusion-view:** keep the `kCall`+attr and have the Stage-1 emitter construct an `HloFusionInstruction` view of `to_apply()` for the tiled emitter.

### 2.2 Intermediates → alloca (the memory-round-trip win)

Once the region is one kernel emitting tensor SSA, the standard CPU bufferization pipeline (`fusion_compiler.cc` `AddBufferizationPasses`: `OneShotBufferize` → `BufferHoisting` → `PromoteBuffersToStack`, `maxAllocSizeInBytes = 4096`) turns non-escaping, statically-sized intermediates into `memref.alloca` automatically — no driver work. This is the lever Stage-0 could not reach and that recovers the ~74%-of-tax memory component. Region-escaping values keep their buffer slices; externally-visible aliasing is unchanged (the region is one opaque kernel to buffer assignment).

### 2.3 The region mechanism is unchanged

`SmallRegionHoistingPass`, its liveness outlining, the cost model (aggregate `bytes_accessed` + min-region-size/control-flow), and the eligibility partition all carry over from Stage-0. Stage-1 changes (i) the executor behind the region and (ii) — once Increment 2 lands — loosens the eligibility list so scatter is no longer a region boundary.

## 3. Key dependency: the experimental tiled emitter / tiling propagation (PARALLEL WORK)

**Stage-1 depends on the experimental tiled-emitter tiling-propagation path, and this dependency is load-bearing.** The CPU tiled emitter's *default* tile-assignment helper (`GetTiling`, `tiled_fusion_emitter.cc:159`) builds a `TileMapping` for the fusion root only. That is sufficient for single-output-domain fusions, but a multi-shape region with a **dot** has hidden tiling parameters (the dot's contracting dimension, `tiling_specification.cc:72`) that the root-only mapping leaves unassigned — emission then fails with `No tile sizes found for instruction: …dot`. The **experimental tiling-propagation path** (`GetTiledHloComputation` / `TilingSpace`, gated today by `--xla_cpu_experimental_enable_tiling_propagation`) assigns tile sizes to *all* tiling-space dimensions and emits dot+reduce+broadcast regions correctly (verified: frag region → one tiled kernel, 7 allocas, correct result).

This path is **experimental and off by default**; productionizing it (making full tiling propagation the standard CPU tile assignment, retiring the "single-tile dummy" placeholder, `tiled_fusion_emitter.cc:339`, b/511084185) is **assumed to be separate, parallel work**. Stage-1's posture toward it:
- **Until it is production:** Stage-1 is itself experimental (behind `xla_cpu_experimental_region_compilation`) and composes the tiling-propagation flag for region-fusions. Dot-containing regions require it; dot-free multi-shape regions (e.g. reduce→broadcast) tile on the default path and work without it.
- **When it lands:** Stage-1 drops the composed flag and relies on the now-default tiling propagation; no other change.
- **Risk shared with that work:** any limitation in tiling propagation (shapes/ops it can't tile, multi-output) directly bounds which regions Stage-1 can fold. The two efforts should share the CPU `IsSupportedInstruction` allow-list and the region benchmark set so a tiling-propagation regression trips the region wire in presubmit.

Alternatively, Stage-1 could carry a CPU-local fix to `GetTiling` to populate the full `TileMapping` from `TilingSpace` for region-fusions, removing the dependency on the experimental flag for the dot case. This is a smaller, region-scoped change and is the fallback if productionizing tiling propagation slips; it is noted as an option, not the primary plan.

## 4. Migration / increments

Each increment adds exactly one capability (per the overall design's "each stage changes one thing"); the genuinely-new code is small because the driver, scatter emitter, and kernel ABI already exist.

- **Increment 1' — straight-line multi-shape region (incl. dot), behind the flag.** Route the region to the tiled emitter (§2.1), loosen the CPU `IsSupportedInstruction` allow-list for dot/reduce/broadcast behind the flag, compose the tiling-propagation path for dot (§3). Decline scatter / control flow / multi-output(tuple-root) regions → fall back to the Stage-0 legacy emitter. **Acceptance:** bitwise-identical output flag on/off on frag + a synthetic reduce→broadcast region; the region emits as ONE tiled kernel (`tiled_emitter` provenance, `memref.alloca` for intermediates); latency ≥ Stage-0, with the alloca win visible on multi-intermediate regions.
- **Increment 2 — scatter inside the region (unblocks jaxley's ~3×).** Loosen the eligibility list so scatter is no longer a region boundary, and emit it inside the tiled region. One open item to confirm at the top of this increment: scatter is MLIR-emittable via `CpuScatterFusion` (a standalone per-fusion emitter); it must be reachable as an `EmitTiledHloInstruction` case (or composed in) for a scatter to live *inside* a tiled region, not just as its own fusion. **Acceptance:** jaxley emits a single (or few) region kernels spanning its scatters; output matches the interpreter; latency approaches the old 373 µs.
- **Increment 3 — control flow.** Lower `kWhile`/`kConditional` inside a region to `scf.while`/`scf.if` so the loop body is one kernel per iteration (loop-carried values, in-place across iterations — highest risk, deferred last).

## 5. Validation

- **Correctness:** bitwise / interpreter-comparison on frag, the synthetic region set, and jaxley (Increment 2). The existing Stage-0 unit tests + the region/token/control-dep guards remain green.
- **Performance:** `bench_hlo` flag on/off on frag, jaxley, mock_mass_matrix; the region MWEs join the in-tree CPU benchmarks (alongside TORAX).
- **Tripwire:** the region benchmark set + the shared `IsSupportedInstruction` allow-list gate both this work and the tiling-propagation productionization (§3).
- **Guardrails:** large models form no multi-instruction regions (cost model), so they are untouched; flag off ⇒ byte-identical Stage-0 path.

## 6. Risks

| risk | status / mitigation |
|---|---|
| Scatter not MLIR-emittable | **Retired** — `CpuScatterFusion`, confirmed live in jaxley kernels. |
| No multi-op/multi-shape driver | **Retired** — `EmitTiledComputation` is exactly it; CPU-wired. |
| Multi-shape incl. dot won't tile as one kernel | **Retired** — verified via tiling propagation (frag → one kernel). Default path needs the dot tile-size plumbing (§3). |
| Intermediates not localized | **Retired** — `PromoteBuffersToStack` makes allocas automatically (verified in dumps). |
| Tiling-propagation productionization slips | Stage-1 composes the experimental flag meanwhile; CPU-local `GetTiling` fix is the fallback (§3). |
| Multi-output (tuple-root) regions | Deferred; v1 targets single-array-root regions (frag qualifies); tiled emitter currently wants a single array root. |
| Scatter reachable only as standalone fusion, not inside a tiled region | Confirm at the top of Increment 2; if so, add an `EmitTiledHloInstruction` scatter case (reusing `CpuScatterFusion`'s `scf.if` bounds-check + reducer emission). |

## 7. Alternatives considered

- **Stage-0 substrate (legacy `ComputationKernelEmitter`):** in production for non-scatter regions; cannot emit scatter (`IrEmitter::HandleScatter` is `Unimplemented`) and gives legacy (not tiled/vectorized) codegen. Kept as the fall-back for ops the tiled path declines.
- **Broaden the elemental partitioner ABI (`computation_partitioner.cc:497`):** wrong model — the elemental emitter is scalar-per-output-element and cannot thread differently-shaped intermediate tensors; also cross-backend blast radius. Rejected.
- **Hand-rolled per-op loop driver:** unnecessary — `EmitTiledComputation` already is the driver.
- **Dispatch fast-path only:** bounded at ~2–3× of a larger gap and cannot remove per-intermediate buffer traffic (overall design §4). Complementary, not sufficient.
