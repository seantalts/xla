# Single-Kernel Region Emission for XLA:CPU

**Status:** Design for review · **Date:** 2026-06-23

## Summary

On the XLA:CPU thunk runtime, each top-level HLO instruction is compiled to its own kernel and executed as its own thunk. For small, loop-structured models (ODE solvers, neuron simulators, MPC, samplers) this granularity is a significant performance regression relative to the older whole-module CPU runtime: a single loop-body iteration becomes dozens of tiny kernels, executed every trip.

Profiling shows the cost is **not** dominated by per-thunk dispatch (measured at ~15–20 ns/thunk). It is dominated by **fine kernel granularity**: every intermediate value round-trips through a buffer-assignment slice in memory instead of staying in a register, and there is no optimization (fusion, vectorization, CSE) across kernel boundaries. The older runtime avoided both by emitting a whole loop body as one inlined, cross-optimized function.

This doc proposes emitting a **region** — a maximal run of adjacent, codegen-compatible instructions — as a **single MLIR kernel** on the existing tiled (xtile) fusion-emitter stack. Differently-shaped intermediates are threaded as tensor SSA values within one kernel; non-escaping intermediates are localized to stack (`alloca`) automatically by the existing bufferization pipeline. A region containing many tiny ops thus compiles to one kernel with intermediates in registers — recovering the older runtime's behavior without resurrecting its architecture.

A region-hoisting HLO pass that partitions a computation's schedule into such regions and outlines each into its own kernel already exists; today it emits each region through a legacy IrEmitter-based path that cannot express several common ops (notably scatter) and produces non-tiled code. This design routes regions to the **tiled emitter** instead, which removes both limitations.

## 1. Background

### 1.1 The regression

Measured on M-series macOS (older whole-module runtime vs. current thunk runtime), small loop-structured workloads regress 1.5–4×. A representative case (a branched-cable neuron simulation, JAX issue #26145): ~28 tiny kernels run per timestep over hundreds of timesteps. Decomposition attributes the regression roughly 3:1 to memory round-trips (kernel granularity) over dispatch. An existing region-hoisting pass that collapses the *non-scatter* runs of such a body into single kernels recovers ~1.3× of this; the remainder requires collapsing *across* scatter and *across differing shapes* into one kernel, which the legacy region emitter cannot do.

### 1.2 What a region is

After HLO optimization (post-fusion, post-library-rewrite, schedule known), a **region** is a maximal schedule-contiguous run of instructions that can all be emitted into one kernel. Region boundaries are instructions that inherently need runtime services — library/custom fusions, custom calls, infeed/outfeed, collectives, and (until the work below extends coverage) scatter, sort, FFT. The region-hoisting pass outlines each maximal eligible run into its own computation and tags it for single-kernel emission; boundary ops remain their own thunks. The pass, its liveness-based outlining, and its cost model (aggregate `bytes_accessed` below a threshold, plus a minimum region size or the presence of control flow) are unchanged by this design — only the **kernel emitter** behind the tag changes.

## 2. Approach: emit the region through the tiled fusion emitter

### 2.1 The tiled emitter already has a region driver

The CPU tiled (xtile) fusion emitter contains exactly the driver a region needs. `EmitTiledComputation` (`xla/codegen/xtile/codegen/fusion_emitter.cc:1034`) walks a computation in schedule order, emits each instruction via `EmitTiledHloInstruction` to a `tensor`-typed SSA value, and threads results through an instruction→value map. It natively emits dot, reduce, broadcast, pad, concatenate, transpose, reshape, and elementwise ops — the op set of the regressing workloads — and intermediates of differing shapes compose naturally because each is a real tensor value rather than a shared per-output-element index domain. CPU already reaches this driver for ordinary fusions via `xla/backends/cpu/codegen/fusion_emitter.cc:284` → `xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.cc:305`.

The work is therefore to route an outlined region to this emitter rather than the legacy one. Two mechanisms are viable; the implementation picks the simpler:
- **Region as fusion:** the hoisting pass emits the region as a `kFusion` whose fused computation is the region, so the existing fusion/tiled path applies unchanged.
- **Fusion view:** keep the region as a tagged call and construct an `HloFusionInstruction` view of its computation for the tiled emitter.

### 2.2 Intermediates become stack allocations automatically

Once a region is a single kernel emitting tensor SSA, the standard CPU bufferization pipeline (`xla/backends/cpu/codegen/fusion_compiler.cc`, `AddBufferizationPasses`: `OneShotBufferize` → `BufferHoisting` → `PromoteBuffersToStack` with `maxAllocSizeInBytes = 4096`) turns non-escaping, statically-sized intermediates into `memref.alloca` with no additional emitter work. This is the lever the legacy region path cannot reach and that recovers the memory-round-trip component of the regression. Region-escaping values keep their buffer slices; externally-visible aliasing is unchanged, since buffer assignment sees the region as one opaque kernel.

### 2.3 Regions must reach the emitter as un-nested ops

The region-hoisting pass runs late in the pipeline — after instruction fusion, layout assignment, and copy insertion — because it needs the final schedule and buffer information. Consequently a hoisted region's body contains the *products* of instruction fusion: nested `kLoop` fusions and materialized constants. The tiled emitter does not accept these — `SymbolicTileAnalysis` rejects nested fusions outright (`xla/codegen/tiling/symbolic_tile_analysis.cc:1221`, "Nested fusions are not supported"), and non-scalar constants are likewise unsupported. The tiled substrate has been validated on the region *shape* (an un-nested `dot → tanh → reduce → divide → broadcast → subtract` chain emits as one tiled kernel with `alloca` intermediates, bitwise-identical to the legacy path and ~35% faster), but a region taken verbatim from the live pipeline declines and falls back to the legacy emitter.

So the region must be presented to the tiled emitter as **un-nested** ops. Two approaches were spiked; the first is chosen.

**Chosen: flatten the region's fused computation (pipeline untouched).** Recursively inline nested `kLoop` fusion bodies into raw ops via `HloInstruction::Defuse()` (`xla/hlo/ir/hlo_instruction.cc:3576`, the structural inverse of fusion — clone the fused root expression into the parent, replace uses), to a fixpoint over the region. Reduce-reducer subcomputations are not fusions and carry through as `to_apply`. Flattening alone is cheap (~half a day) and validated: a hand-flattened live-frag region emits as one tiled kernel, bitwise-identical to legacy.

But flattening is **necessary, not sufficient** — the spike found a second, independent blocker: **non-scalar constants**. A live region materializes its constants internally (e.g. the frag's twenty `f32[16,16]` dot weights). The tiled emitter rejects non-scalar constants (`xla/codegen/xtile/codegen/fusion_emitter.cc:955`; CPU allow-list `tiled_fusion_emitter.cc:186`), and — more fundamentally — the kernel ABI builds argument buffers strictly from the region call's *operands* (`xla/codegen/emitters/kernel_api_builder.cc:397`), so a region-internal constant's buffer slice never reaches the kernel. **Resolution: lift region-internal non-scalar constants to region-call operands** (they already have buffer slices), via a small HLO rewrite at or before hoisting — i.e. *before* buffer assignment, so the slices exist for the new operands. (This cannot live in the emit-time fusion-view builder, which runs post-buffer-assignment.) Empirically, flatten + constants-as-operands ⇒ one tiled kernel, correct; constants-internal ⇒ declines. Estimated total ~3–5 days (flattening small; constant-lifting is the bulk).

**Rejected: hoist before instruction fusion.** Moving the hoisting pass ahead of instruction fusion does not work on its own: instruction fusion runs over *all* non-fusion computations including the region's callee (`InstructionFusion`/`MakeNonfusionComputations`), so it simply re-fuses the hoisted region body into nested `kLoop` fusions again (probe-confirmed: the region still held ~79 nested fusions and still declined). Making it work would additionally require fencing region computations from instruction fusion, blocking the single-call-site `CallInliner` (active under fast-compile) from inlining region calls back, and re-tuning the cost model for unfused (more numerous, constant-heavy) granularity — ~1–2 weeks, touching shared backend-agnostic passes. Higher cost and risk than flattening, which leaves the pipeline untouched.

Either way this is graph-shaping in front of a validated substrate, not a limitation of the substrate.

### 2.4 Why not the elemental emitter

XLA has a second MLIR emission model (the elemental / loop-fusion emitter, via `computation_partitioner.cc` and `elemental_hlo_to_mlir.cc`) that computes each root output element from the computation's parameters. It is scalar-per-output-element and shares a single output-index domain across all ops, so it cannot thread differently-shaped intermediate tensors — it is not a viable substrate for a multi-shape region, and broadening its function-ABI gate (`computation_partitioner.cc:497`) would not change that. The tiled emitter is the correct substrate.

## 3. Dependency: full tiling propagation for multi-shape regions

This design depends on **tiling propagation** in the CPU tiled emitter, and the dependency is load-bearing for any region containing a dot.

The tiled emitter must assign tile sizes to every dimension in a fusion's tiling space. The default CPU tile-assignment helper (`GetTiling`, `tiled_fusion_emitter.cc:159`) builds a tile mapping for the fusion root only. That suffices for single-output-domain fusions, but a multi-shape region containing a **dot** has hidden tiling parameters — the dot's contracting dimension (`tiling_specification.cc:72`) — that a root-only mapping leaves unassigned; emission then fails (`No tile sizes found for instruction: …dot`). The tiling-propagation path (`GetTiledHloComputation` / `TilingSpace`) assigns tile sizes to *all* tiling-space dimensions and emits dot+reduce+broadcast regions correctly.

Tiling propagation is currently experimental and off by default (gated by `--xla_cpu_experimental_enable_tiling_propagation`; the default path uses a placeholder single-tile strategy). **Productionizing it — making full tiling propagation the standard CPU tile assignment — is assumed to be separate, ongoing work, and is a prerequisite for the dot-containing regions that matter most here.** This design's relationship to it:

- **While experimental:** region emission stays behind its own experimental flag and composes the tiling-propagation flag for region fusions. Dot-free multi-shape regions (e.g. reduce → broadcast) tile on the default path and do not require it; dot-containing regions do.
- **Once production:** region emission drops the composed flag and relies on the now-default tiling propagation; no other change.
- **Shared risk surface:** any op or shape that tiling propagation cannot tile directly bounds which regions can be folded. The two efforts should share the tiled emitter's instruction allow-list and a common benchmark set so a tiling-propagation regression is caught against region emission in presubmit.
- **Fallback if it slips:** a region-scoped fix to `GetTiling` that populates the full tile mapping from the tiling space for region fusions removes the dependency on the experimental flag for the dot case. This is a smaller, contained change, noted as a fallback rather than the primary plan.

The other capability this design relies on — emitting scatter through MLIR — already exists: scatter is emitted by a dedicated MLIR emitter (`CpuScatterFusion`, `xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.cc`), routed at `xla/service/cpu/thunk_emitter.cc` for scatter-rooted fusions. (The legacy IrEmitter's `HandleScatter` is unimplemented, which is why the legacy region path cannot collapse across scatter.)

## 4. Implementation plan

Each milestone adds one capability. The driver, scatter emitter, and kernel ABI already exist, so the net new code is small.

### Milestone 1 — straight-line multi-shape regions (pointwise, dot, reduce, broadcast)

Route a region with no scatter and no control flow to the tiled emitter, behind an experimental flag. Loosen the CPU tiled-emitter instruction allow-list (`IsSupportedInstruction`, `tiled_fusion_emitter.cc:177`, currently `default: IsElementwise()`) to admit dot/reduce/broadcast when the region path is active, and compose tiling propagation for the dot case (§3). Present the region to the emitter as un-nested ops (§2.3) — without this step, live post-fusion regions decline. Regions containing scatter, control flow, or a tuple (multi-output) root decline and fall back to the legacy region emitter. Flag off ⇒ the existing legacy path is byte-for-byte unchanged.

**Status: implemented and verified.** The tiled-emitter routing, the allow-list loosening (behind an `admit_region_ops` flag, default off), declines-with-fallback, the fusion-view of the region's computation, `Defuse()`-based flattening (§2.3), and non-scalar-constant lifting (§2.3, via an `exclude_nonscalar_constants` option on the hoisting pass wired from the region flag) are all in place. A representative live region (the frag entry: dot/tanh/reduce/divide/broadcast/subtract with twenty `f32[16,16]` weight constants) now folds into **one tiled region kernel** under the flag (`tiled_emitter` provenance, stack-allocated intermediates; the lifted constants become call operands), and remains the legacy single kernel with the flag off (byte-identical). Correctness checked on a synthetic frag-shaped region with non-zero constants: bitwise-identical flag on/off for a non-cancelling root; for a mean-subtraction root the results differ only at ~1e-9 (floating-point reassociation on a mathematically-zero quantity — benign). Note: a single straight-line region like the frag already collapses to one kernel on the legacy path, so latency here is at parity (floor-dominated); the *latency* win this work targets appears on workloads with many regions / per-iteration loop bodies, which require scatter (next milestone) to fold.

**Acceptance:** bitwise-identical results flag on/off on representative regions (e.g. a `dot → tanh → reduce → divide → broadcast → subtract` chain and a `reduce → broadcast` chain); the region emits as a single tiled kernel (tiled-emitter provenance, `memref.alloca` for intermediates); latency no worse than the legacy region path, with the stack-localization win visible on multi-intermediate regions.

### Milestone 2 — scatter inside a region

Stop treating scatter as a region boundary and emit it inside the tiled region. One item to confirm at the start of this milestone: scatter is emittable via the standalone `CpuScatterFusion`; it must be reachable as an `EmitTiledHloInstruction` case (reusing that emitter's bounds-checked store + reducer emission) so a scatter can live *inside* a tiled region rather than only as its own fusion. This is the milestone that addresses the largest remaining share of the regression for scatter-heavy workloads.

**Acceptance:** scatter-containing loop bodies (e.g. the #26145 neuron simulation) emit as one or few region kernels spanning their scatters; results match the interpreter; latency approaches the older runtime.

### Milestone 3 — control flow inside a region

Lower `while`/`conditional` inside a region to `scf.while`/`scf.if`, so a loop body is one kernel per iteration rather than a thunk graph re-executed each trip. Highest risk (loop-carried values, in-place updates across iterations); sequenced last.

## 5. Validation

- **Correctness:** bitwise / interpreter comparison on the representative regions, scatter workloads (Milestone 2), and control-flow workloads (Milestone 3). Existing region-hoisting unit tests (including token-ordering and control-dependency guards) stay green.
- **Performance:** `bench_hlo` with the region flag on/off on the regressing workloads; the region MWEs join the in-tree CPU benchmark set.
- **No-regression guardrail:** the cost model forms multi-instruction regions only where work is small and dispatch dominates, so large models form no such regions and are untouched. Flag off ⇒ unchanged behavior.
- **Shared tripwire:** the region benchmark set and the tiled-emitter instruction allow-list gate both this work and the tiling-propagation productionization (§3).

## 6. Alternatives considered

- **Legacy IrEmitter region path (the current emitter behind the region tag):** in production for non-scatter regions, but cannot emit scatter (`HandleScatter` unimplemented) and produces non-tiled code. Retained as the fallback for ops the tiled path declines.
- **Broadening the elemental partitioner ABI:** wrong substrate — the elemental emitter is scalar-per-output-element and cannot thread differently-shaped intermediate tensors (§2.3); also has cross-backend blast radius. Rejected.
- **A hand-written per-op loop driver:** unnecessary — `EmitTiledComputation` already is that driver.
- **Dispatch fast-path only (pre-resolved pointers, lighter executor):** addresses only the ~quarter of the cost that is dispatch and cannot remove per-intermediate buffer traffic. Complementary, not sufficient.

## Appendix: key code references

- Tiled region driver: `xla/codegen/xtile/codegen/fusion_emitter.cc:1034` (`EmitTiledComputation`), per-op tensor emission `:447,:887` (`EmitTiledHloInstruction`).
- CPU tiled entry + gates: `xla/backends/cpu/codegen/fusion_emitter.cc:284`; `xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.cc:177` (`IsSupportedInstruction`), `:159` (`GetTiling`, root-only tile mapping), `:305` (`EmitTiledFusionKernel`), `:339` (placeholder tile strategy).
- Tiling space / propagation: `xla/codegen/tiling/symbolic_tile_analysis.cc` (handles dot), `xla/codegen/tiling/tiling_specification.cc:72` (hidden tiling parameter).
- Scatter (MLIR): `xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.cc`; routed at `xla/service/cpu/thunk_emitter.cc`.
- Bufferization → stack: `xla/backends/cpu/codegen/fusion_compiler.cc` (`AddBufferizationPasses`, `PromoteBuffersToStack`).
- Elemental (non-substrate) model: `xla/codegen/emitters/computation_partitioner.cc:497`, `xla/codegen/emitters/elemental_hlo_to_mlir.cc`.
