# CPU Priority Fusion — Design (v6)

**Status:** Draft (v6, refactor restructured into in-place decouple + mechanical relocation)
**Date:** 2026-04-21
**Author:** `claude/gpu-priority-fusion-xla-kBHi3`
**Supersedes:** v1–v5 (see git log). v6 splits the refactor into Phase 0 (in-place decouple at `xla/backends/gpu/transforms/`) and Phase A (pure file relocation). This removes intermediate states where generic code would depend on GPU code and makes each decoupling step an independently committable, testable change.

## Problem

XLA's GPU backend has a `PriorityFusion` pass that greedily fuses producer/consumer pairs ranked by a cost-model-estimated runtime benefit. XLA:CPU has only `CpuInstructionFusion`, which uses static opcode heuristics and a handful of empirically-tuned guards. CPU has no fusion-time cost model.

Bringing priority-style fusion to CPU should improve fusion quality while keeping the hard lessons of the existing CPU guards. The GPU pass's *outer driver* is backend-neutral, but the real work lives in a nested `PriorityFusionQueue` class (~850 lines) that directly owns GPU-specific state: `GpuHloCostAnalysis`, `GpuPerformanceModelCache`, `GpuPerformanceModel`, `HloFusionAnalysisCache`, `mlir::MLIRContext*`, and `FusionDeduplicationCache`. Any shared-location refactor has to move the queue too and thread those dependencies behind an interface.

## Goals

1. Move `PriorityFusion` **and** `PriorityFusionQueue` to a hardware-independent location under `xla/hlo/transforms/`.
2. Parameterize behind a small `FusionCostModel` interface (6 methods).
3. Relocate GPU-specific state (`GpuPerformanceModelCache`, `FusionDeduplicationCache`, `HloFusionAnalysisCache`, `mlir::MLIRContext*`) into `GpuFusionCostModel` so the shared queue holds no GPU types.
4. Add a `CpuPriorityFusion` subclass + `CpuFusionCostModel` implementation.
5. Add a guard rule: for the hypothetical fused candidate, refuse fusion when `kflops > X` AND `independent_work_items < 32`.
6. Treat `dynamic-update-slice`, `dynamic-slice`, `concatenate`, `pad` as "complex" ops with inflated per-element kflops so the guard naturally catches pathological CPU codegen.
7. Virtualize `IsFusible` on the base pass so CPU can reject ops its emitter can't lower (scatter, reduce-window, gather). Also reject non-scalar constant producers on CPU to match today's `CpuInstructionFusion` behavior.
8. `CpuFusionCostModel::WouldExplodeIrSize` wraps the **existing CPU cap**, `FusionNodeIndexingEvaluation::CodeDuplicationTooHigh` (`cpu_instruction_fusion.cc:439`), not the 16 KB `kFusionThresholdBytes` (which is only used inside the kDot branch).
9. Ship behind `xla_cpu_use_priority_fusion` (default off); zero behavior change for GPU.

## Non-goals

- Relaxing the existing `CpuInstructionFusion` caps (`CodeDuplicationTooHigh`, 5-reduction cap, concat-on-minor-dim hard block, 8-operand concat cap). The new pass preserves them. A follow-up CL may loosen them.
- Vectorization-aware cost modeling. Structured to allow later addition as a pure internal change to `CpuFusionCostModel`; not in this CL.
- Removing `CpuInstructionFusion`. Stays as the default path.
- Adding the guard to GPU. Rule lives in `CpuPriorityFusion` only.
- Per-opcode latency tables for CPU beyond the four complex ops.
- A CPU analog of GPU's `FusionFitsInBudget`. LLVM IR parameter-count caps for many-operand fusions may be needed before the flag flips; **tracked as an open item**, not implemented here.


## Architecture

### Phasing

The refactor is split into two stages to keep each change individually buildable and testable:

**Phase 0 — in-place decouple.** All the risky refactoring (splitting `PriorityFusionQueue`'s GPU state into `GpuFusionCostModel`, introducing the `FusionCostModel` interface, hoisting base-class virtuals, creating `GpuPriorityFusion`) happens **without moving any files**. Base pass + queue stay at `xla/backends/gpu/transforms/priority_fusion.{h,cc}` (namespace `xla::gpu`) throughout. The shared interface lands at its final location (`xla/hlo/transforms/fusion_cost_model.h`) from day one because it has no dependencies, but the pass files don't move. Each Phase 0 task is its own commit that passes the full GPU test suite before and after.

**Phase A — file relocation.** Purely mechanical. `git mv` the now-decoupled base pass + queue to `xla/hlo/transforms/`, change `namespace xla::gpu` → `namespace xla`, update include guards and BUILD deps. At this point no code changes logic — just paths and namespaces. A single atomic commit is fine because the changes are shallow.

**Phases B–E** add CPU code on top of the cleanly relocated shared pass.

This split means generic code never depends on GPU code at any intermediate point — the in-place refactor removes all GPU coupling before any move. If Phase A surfaces a residual GPU reference in the supposedly-generic file, that's a Phase 0 bug and the move can be reverted cheaply.

### Three-layer split (final state)

```
xla/hlo/transforms/
  fusion_cost_model.h             ← FusionCostModel interface (6 methods)
  priority_fusion.{h,cc}          ← MOVED from xla/backends/gpu/transforms/
                                    Includes PriorityFusionQueue. No GPU types.
  priority_fusion_test.cc         ← MOVED, plus a FakeCostModel-driven test.

xla/backends/gpu/transforms/
  gpu_priority_fusion.{h,cc}      ← NEW thin subclass. Same ctor signature as
                                    old PriorityFusion, so call-site churn is
                                    a type rename.
  gpu_fusion_cost_model.{h,cc}    ← NEW. Owns GpuHloCostAnalysis,
                                    GpuPerformanceModel, GpuPerformanceModelCache,
                                    HloFusionAnalysisCache, MLIRContext,
                                    FusionDeduplicationCache, tiled run-time
                                    data cache. Implements the 6-method interface.

xla/service/cpu/
  cpu_priority_fusion.{h,cc}      ← NEW subclass. Guard + concat-on-minor-dim
                                    pre-check + IsFusible override.
  cpu_fusion_cost_model.{h,cc}    ← NEW. Wraps HloCostAnalysis +
                                    FusionNodeIndexingEvaluation.
```

**Responsibilities:**

- **Shared base pass + queue** (`xla/hlo/transforms/priority_fusion.{h,cc}`): owns the fusion loop, the priority queue, the `can_fuse_cache_` (a cache of the backend legality functor's results), and the `Fuse()` mechanics. Depends only on `FusionCostModel` + the pass's virtual hooks; contains zero GPU types.
- **`CanFuse` ordering** in the queue: (1) root guard, (2) `IsFusible(producer)`/`IsFusible(consumer)` via pass virtual, (3) `backend_can_fuse_(p, c)` — backend checks including bitcast-consumer and any backend-specific ordering, (4) `WouldExplodeIrSize` via cost model, (5) `InstructionFusion::ShouldFuseInPlaceOp(p, c, alias_info, std::nullopt)` as the final generic aliasing/in-place safety check. Steps 1, 2, 4, 5 are in the base pass; step 3 is fully delegated so each backend owns its internal ordering (GPU: Triton first, then bitcast, then scatter/reduce/budget — matches today exactly; CPU: concat hard-block, bitcast, kflops/work-items guard).
- **Backend subclasses** override three protected virtuals:
  - `BackendCanFuse(producer, consumer)` — backend-specific legality layered on top of shared checks. GPU: Triton preference, bitcast-consumer forbid (kept at original post-Triton position so Triton still gets first look), `CanEmitInputFusedScatter`, reduce-into-reduce, reduce→loop switch avoidance, `FusionFitsInBudget`. CPU: bitcast-consumer forbid, `kflops/work_items` guard, concat-on-minor-dim hard block.
  - `ChooseKind(producer, consumer)` — `FusionKind` selection.
  - `IsFusible(instruction)` — opcode-level fusibility. Default matches the current GPU allowlist; CPU overrides to reject scatter/gather/reduce-window and non-scalar constant producers, matching `CpuInstructionFusion::CanBeLoopFused` + the existing non-scalar-constant guard at `cpu_instruction_fusion.cc:339–342`.
- **Backend cost models** implement the 6-method interface and own all backend-specific analysis caches.

### The `FusionCostModel` interface

```cpp
// xla/hlo/transforms/fusion_cost_model.h
class FusionCostModel {
 public:
  struct RunTimes {
    absl::Duration unfused;  // producer runs once + each consumer runs once
    absl::Duration fused;    // producer inlined once per consumer
  };

  virtual ~FusionCostModel() = default;

  // Called once per computation before any EstimateRunTimes calls. GPU
  // uses this to run its initial cost-analysis Accept() over the
  // computation and populate internal caches. CPU uses it to Accept()
  // the shared HloCostAnalysis. Default: ok, no-op.
  virtual absl::Status Prepare(HloComputation* computation) {
    return absl::OkStatus();
  }

  // The base pass does `priority = unfused - fused`, plus incremental
  // arithmetic when the consumer set changes. Returning the pair (rather
  // than a single Duration) preserves GPU's existing incremental math
  // inside CalculateProducerPriority.
  virtual RunTimes EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<HloInstruction* const> consumers) = 0;

  // True if fusing `producer` into `consumer` would produce code
  // exceeding the backend's code-size budget. GPU wraps
  // GpuHloCostAnalysis::ProducerConsumerMergedTooLarge. CPU wraps
  // FusionNodeIndexingEvaluation::CodeDuplicationTooHigh.
  virtual bool WouldExplodeIrSize(const HloInstruction* producer,
                                  const HloInstruction* consumer) = 0;

  // Called by the base pass after a fusion is performed. Lets the cost
  // model invalidate caches keyed on pre-fusion instructions. CPU uses
  // it to invalidate its FusionNodeIndexingEvaluation for `consumer`.
  // Default: no-op.
  virtual void OnInstructionFused(HloInstruction* producer,
                                  HloInstruction* consumer,
                                  HloInstruction* fusion) {}

  // Called on instructions whose inputs or users changed and need
  // re-costing. GPU invalidates `GpuPerformanceModelCache` entries here.
  // Default: no-op.
  virtual void Invalidate(const HloInstruction* instruction) {}

  // Called to revisit an instruction's underlying cost analysis
  // (HloCostAnalysis::RevisitInstruction). Default: ok, no-op.
  virtual absl::Status Revisit(const HloInstruction* instruction) {
    return absl::OkStatus();
  }
};
```

**Why 6 methods:** The base pass calls into the cost model from five places: initial per-computation setup (`Prepare`), priority computation (`EstimateRunTimes`), IR-size check (`WouldExplodeIrSize`), post-fusion cache invalidation (`OnInstructionFused`, `Invalidate`), and `RevisitInstruction` after fusion. `Prepare` was added in v3 after v2 reviewer noted the original GPU `PriorityFusionQueue` constructor did this work implicitly; after the queue move, it has no caller without an explicit hook.

**Bitcast / constant priority short-circuits stay in the base pass.** The current GPU code gives bitcasts `absl::InfiniteDuration()` and constants `-absl::InfiniteDuration()`. These are not cost-model concerns — they're queue-ordering conventions. The base pass keeps them as constants in `ComputeAndSetPriorities`.

### `PriorityFusionQueue` changes

The queue moves to the shared file and loses its GPU state. Specifically:

- Removed from queue: `GpuHloCostAnalysis cost_analysis_`, `GpuPerformanceModelCache gpu_performance_model_cache_`, `GpuPerformanceModel gpu_performance_model_`, `HloFusionAnalysisCache& fusion_analysis_cache_`, `mlir::MLIRContext* mlir_context_`, `FusionDeduplicationCache fusion_deduplication_cache_`, `tiled_run_time_data_cache_`.
- Moved to `GpuFusionCostModel`: all of the above plus the GPU-specific cache of tiled run-time data.
- Kept in queue: `can_fuse_cache_` (generic cache of backend legality functor's results), the `std::map<pair<Duration, int64_t>, HloInstruction*>` priority queue itself, `operands_to_new_consumers_`, `operands_to_removed_consumers_runtimes_` (generic, its payload is the new `RunTimes` struct), and the `reverse_map_` of instructions to keys.

**Incremental priority math.** `CalculateProducerPriority` preserves the existing delta logic: supplies only the **changed** consumer set (`operands_to_new_consumers_`) to `EstimateRunTimes`, subtracts the cached `RunTimes` of removed consumers held in `operands_to_removed_consumers_runtimes_`, adds to the previously cached `current_priority`, and updates the cache. This keeps per-update work proportional to changed consumers, not to `|users()|`. Without retaining `operands_to_removed_consumers_runtimes_` the cost would degenerate to a full-users recomputation per update — O(N²) compile-time. See plan Task A2 Step 5 for the exact code path.

**Constructor:** The queue takes `FusionCostModel* cost_model` and `absl::AnyInvocable<FusionDecision(HloInstruction*, HloInstruction*)> backend_can_fuse` — passed by the pass as a bound reference to its own `BackendCanFuse` virtual. This sidesteps the "nested class can't call a pass virtual" problem without hoisting `CanFuse` out of the queue.

### Base pass constructor and ownership

```cpp
class PriorityFusion : public HloModulePass {
 public:
  // Takes ownership of cost_model. Subclasses build it before calling
  // this constructor; no post-hoc setter, no nullable cost_model_.
  PriorityFusion(tsl::thread::ThreadPool* thread_pool,
                 const AliasInfo* alias_info,
                 std::unique_ptr<FusionCostModel> cost_model);
  // ...
 protected:
  virtual FusionDecision BackendCanFuse(HloInstruction* p, HloInstruction* c) {
    return FusionDecision::Allow();
  }
  virtual HloInstruction::FusionKind ChooseKind(const HloInstruction* p,
                                                 const HloInstruction* c) {
    return HloInstruction::FusionKind::kLoop;
  }
  virtual bool IsFusible(const HloInstruction& instr);  // impl in base, virtual for CPU override

  FusionCostModel* cost_model() { return cost_model_.get(); }
  // ...
 private:
  std::unique_ptr<FusionCostModel> cost_model_;
  // ...
};
```

### `tiled_run_time_data_cache_` and `GetTiledRunTimeDataCached`

Today these live in `PriorityFusionQueue` and are consulted inside `CalculateProducerPriority` for Triton tiling cost estimation. In v3 this call path is **fully folded into `GpuFusionCostModel::EstimateRunTimes`** — the cache and the helper are private members/methods of `GpuFusionCostModel`, and the shared queue never touches them. No interface expansion needed; GPU-specific types (`TiledRunTimeDataOrError`, `BlockLevelParameters`) stay internal to the GPU implementation.

### GPU subclass (refactor only)

- `GpuFusionCostModel` owns `GpuHloCostAnalysis`, `GpuPerformanceModel`, `GpuPerformanceModelCache`, `HloFusionAnalysisCache`, `mlir::MLIRContext*`, `FusionDeduplicationCache`. Its `EstimateRunTimes` pastes the existing GPU priority-computation logic verbatim. `WouldExplodeIrSize` forwards to `ProducerConsumerMergedTooLarge`. `OnInstructionFused` updates `GpuPerformanceModelCache`. `Invalidate` clears cache entries.
- `GpuPriorityFusion::BackendCanFuse` pastes the current `CanFuse` backend blocks in their original order: Triton preference first, then `IsFusibleBitcast(consumer)` forbid (preserved in original post-Triton position), then `CanEmitInputFusedScatter`, significant-reduce-into-reduce, reduce→loop switch, `FusionFitsInBudget`. Ordering matters for GPU semantics (Triton gets first chance at bitcast-consumer cases).
- `GpuPriorityFusion::IsFusible` returns the current GPU allowlist.
- **No behavior change.** The entire GPU priority-fusion test suite must pass unchanged.

### CPU subclass (new)

#### `CpuFusionCostModel`

Constructor takes:
- `HloCostAnalysis*` (injected; constructed by `CpuPriorityFusion`. **Non-const** so `Prepare()` can call `computation->Accept(analysis)` during setup.)
- `const CpuDeviceInfo&` (num threads, peak kflops/sec, peak mem bandwidth — enough to plug in vectorization later)
- `CpuFusionCostModelOptions` (complex-op penalty table)

Implements the interface:

- **`EstimateRunTimes(producer, consumers)`**: computes `unfused = time(producer) + sum(time(consumer_i))` and `fused = sum(time(producer + consumer_i))` using `time(op) = max(kflops/peak, bytes/bandwidth)` as a simple overlap proxy. The max-based formula is not cycle-accurate; it's sufficient for ranking.
- **`WouldExplodeIrSize(producer, consumer)`**: wraps `FusionNodeIndexingEvaluation::CodeDuplicationTooHigh(producer)` — the existing CPU cap — **not** the 16 KB threshold. Also preserves the 5-reduction cap from `cpu_instruction_fusion.cc:392–410`.
- **`OnInstructionFused` / `Invalidate` / `Revisit`**: default no-ops. Future additions if profiling shows a need.

Plus CPU-only helpers used by `CpuPriorityFusion::BackendCanFuse`:

- `Kflops(instr)`: ordinary ops → `hlo_cost_analysis_->flop_count(instr) / 1000`. Complex ops → `num_elements * per_elem_penalty / 1000` from the options struct.
- `IndependentWorkItems(instr)`: **physical** outermost dim trip count — `shape.dimensions(LayoutUtil::Major(shape.layout(), 0))`. Guarded by `shape.has_layout()`; falls back to `shape.dimensions(0)` (logical first dim) when no layout. For tuples: takes the **first** element's work items (not min — avoids spurious refusals for multi-output fusions with scalar secondary outputs). For scalars: 1.
- `FusedKflops(producer, consumer)`: `Kflops(producer) + Kflops(consumer)`. One fused instance — per-pair.
- `FusedWorkItems(producer, consumer)`: `IndependentWorkItems(consumer)`.

```cpp
struct ComplexOpPenalties {
  int64_t concatenate_kflops_per_element = 4;
  int64_t pad_kflops_per_element = 2;
  int64_t dynamic_slice_kflops_per_element = 2;
  int64_t dynamic_update_slice_kflops_per_element = 3;
  // concat_minor_dim_multiplier removed in v2: the hard block in
  // BackendCanFuse makes it unreachable today. Re-add if the hard
  // block is loosened in a follow-up CL.
};
```

#### `CpuPriorityFusion`

Overrides three virtuals:

1. `IsFusible(instr)` — CPU allowlist matching `CpuInstructionFusion::CanBeLoopFused`: `kElementWise`, `kBitcast`, `kBroadcast`, `kConcatenate`, `kDynamicSlice`, `kDynamicUpdateSlice`, `kIota`, `kPad`, `kReduce`, `kReshape`, `kReverse`, `kSlice`, `kTranspose`. Rejects `kGather`, `kScatter`, `kReduceWindow`. Additionally rejects non-scalar `kConstant` producers — preserves the existing guard at `cpu_instruction_fusion.cc:339–342` that prevents 2-instruction fusions containing only a constant and another node.
2. `ChooseKind(p, c)` — default `kLoop`. Future: detect in-place DUS cases.
3. `BackendCanFuse(p, c)`:
   1. Bitcast-consumer forbid: `if (IsFusibleBitcast(*c)) return FusionDecision::Forbid("not fusing into a single bitcast as consumer");` (moved here so CPU owns its ordering; mirrors GPU).
   2. Concat-on-minor-dim hard block, verbatim from `cpu_instruction_fusion.cc:358–380`. Guarded by `shape.has_layout()`.
   3. `>8-operand concat` block, verbatim.
   4. **The guard**: `int64_t kflops = cpu_cost_model_->FusedKflops(*p, *c); int64_t wi = cpu_cost_model_->FusedWorkItems(*p, *c); if (kflops > options_.kflops_refuse_threshold && wi < options_.min_work_items) return FusionDecision::Forbid("kflops/work-items guard");`
   5. Return `Allow()`.

**Guard semantics note.** The guard is per-pair. The base pass only fuses a producer into a consumer if it can fuse into *all* non-bitcast users (`CanFuseWithAllNonBitcastUsers`). So a single per-pair refusal aborts the producer's entire fan-out fusion — fusion happens all-or-nothing for a given producer. This is intentional: if *any* consumer would create a pathological fused kernel, we keep the producer materialized as a separate buffer. If we later want per-user decisions, the base pass (not the guard) would need to change.

```cpp
struct CpuPriorityFusionOptions {
  int64_t kflops_refuse_threshold = 10'000;   // 10 Mflops — first guess
  int64_t min_work_items = 32;                 // first guess
  ComplexOpPenalties complex_op_penalties;
  CpuDeviceInfo device_info;

  // Compile-time controls (safety valves; defaults don't limit quality).

  // Hard cap on fusions per computation. 0 = unlimited.
  int64_t max_fusions_per_computation = 0;

  // Precise invalidation of FusionNodeIndexingEvaluation: only evict
  // the consumer whose fusion grew. Set false only for diagnosis.
  // Plumbed into CpuFusionCostModel via its constructor.
  bool precise_evaluation_invalidation = true;

  // Thread pool for parallel initial priority computation, matching
  // GPU's pattern. nullptr = sequential. Passed straight through to
  // PriorityFusion base ctor.
  tsl::thread::ThreadPool* thread_pool = nullptr;
};
```

**Compile-time context.** `CpuPriorityFusion` is expected to be 5–20× slower than `CpuInstructionFusion` in the fusion phase — mostly from `FusionNodeIndexingEvaluation` reconstruction. The default `precise_evaluation_invalidation = true` brings this back to approximately linear. `thread_pool` parallelizes initial priority computation.

**Deferred:** `wall_time_budget` is intentionally not in this CL. Wiring it correctly requires base-pass changes (a deadline check in the queue loop) that are out of scope; add in a follow-up when benchmarks show we actually need it.

**Plausibility check on thresholds.** With default penalties: a minor-dim concat of a `f32[128,128]` is `16384 * 4 * 1000⁻¹ = 65 kflops` — above `X=10_000`? No, `65 < 10_000` in kflops. The threshold `X = 10'000` kflops = **10 Mflops** is actually quite high; `f32[128,128]` concat trips it only if the number of elements × penalty exceeds 10M. Example concat that trips: `f32[4096, 4096]` concatenate on minor dim → `16M * 4 kflops/elem = 64M kflops ≫ 10M` → trips if fan-out also drops work items below 32. The `(X, 32)` pair is thus a **"very large complex op being fused into a thin-outer-dim consumer"** filter. Benchmarks during rollout will retune; expose as plain fields so tuning is config, not code.

### Pipeline integration

`xla/service/cpu/cpu_compiler.cc` around line 1053:

```cpp
AliasInfo alias_info;
bool use_multi_output_fusion = options::UseMultiOutputFusion(module->config());
if (module->config().debug_options().xla_cpu_use_priority_fusion()) {
  pipeline.AddPass<CpuPriorityFusion>(
      &alias_info, ShapeSizeBytesFunction(),
      CpuPriorityFusionOptions{/* defaults */});
} else {
  pipeline.AddPass<CpuInstructionFusion>(
      &alias_info, /*may_duplicate=*/!use_multi_output_fusion);
}
```

**Flag:** `bool xla_cpu_use_priority_fusion = 503;` in `DebugOptions` (`xla/xla.proto`). Default `false`.

**Rollout:**
1. Land pass behind flag, default off.
2. Internal benchmarks on representative CPU model suite.
3. Address any `FusionFitsInBudget`-equivalent needs (LLVM IR parameter-count blowup) before flipping.
4. Flip default on for a subset of targets (e.g., `x86_64`).
5. Remove `CpuInstructionFusion` + flag in a later CL.

### GPU call-site updates

Four files, type rename only (constructor signature unchanged):

- `xla/service/gpu/fusion_pipeline.cc:76`
- `xla/backends/gpu/transforms/triton_fusion_numerics_verifier.cc:~136`
- `xla/backends/gpu/autotuner/triton.cc:~276`
- `xla/backends/gpu/autotuner/fission_backend.cc:~127`

Plus each file's BUILD dep updates from `//xla/backends/gpu/transforms:priority_fusion` to `//xla/backends/gpu/transforms:gpu_priority_fusion`.


## Testing

### Base pass (zero behavior change)

- Move `priority_fusion_test.cc` alongside the new location. Existing tests must pass with `GpuPriorityFusion` subclass.
- Add one new test using a `FakeCostModel` (constant `RunTimes`, `false` `WouldExplodeIrSize`) to exercise the base pass's queue mechanics without any backend.

### CPU-specific (`cpu_priority_fusion_test.cc`)

1. **Guard fires on complex-op fusion with low work items.**
2. **Guard does not fire when work items ≥ 32.**
3. **Guard does not fire when kflops ≤ X.**
4. **One test per complex op** (DUS, dynamic-slice, concatenate, pad).
5. **Priority ordering sanity.** Cheap elementwise vs complex-op producer into the same consumer ⇒ cheap one picked first.
6. **Flag-off smoke test.** With `xla_cpu_use_priority_fusion=false`, the `CpuInstructionFusion` branch is taken. Assert the branch selected, not HloProto equality (which would be vacuous — the default path is unchanged).
7. **`IsFusible` rejects scatter/reduce-window/gather on CPU.**
8. **Layout-sensitive paths.** Module without layouts: `BackendCanFuse` must not CHECK-fail. Module with layouts: minor-dim concat is hard-blocked; major-dim concat allowed.

### CPU cost model (`cpu_fusion_cost_model_test.cc`)

- `Kflops`, `IndependentWorkItems`, `FusedKflops`, `FusedWorkItems` on small synthetic HLOs with known-correct values.
- `EstimateRunTimes` returns `{unfused, fused}` where `unfused > fused` for "obviously good" fusions and `unfused < fused` for "obviously bad" ones.
- `WouldExplodeIrSize` matches `CpuInstructionFusion`'s `CodeDuplicationTooHigh` decision on a set of paired HLOs.
- `IndependentWorkItems` returns the **physical** outermost dim for transposed layouts (`f32[1024,4]{0,1}` returns 4, not 1024).

### GPU regression gate

Existing `priority_fusion_test.cc` under the new path. Any drift blocks the refactor.

### End-to-end

Two small CPU end-to-end runs: flag-on numerical correctness (rtol=1e-5, atol=1e-6) against flag-off baseline, compilation smoke-test for both. Record fusion count delta informationally.

## Risks & open questions

- **Threshold values** (`X = 10'000 kflops`, `work_items = 32`, complex-op penalties): first guesses. Re-tuned during benchmark rollout; structured as a plain-field options struct.
- **`IndependentWorkItems` uses physical outermost dim** (via `LayoutUtil::Major`). Assumes the CPU emitter parallelizes that dim. If the emitter's choice differs (e.g., it parallelizes a different outer loop), the metric will be systematically off. Pre-layout HLO falls back to logical `dimensions(0)`.
- **Per-pair guard aborts whole-producer fusion.** Intentional: any pathological pair vetoes the producer's entire fan-out. If a later CL wants per-user fusion granularity, the base pass's `CanFuseWithAllNonBitcastUsers` behavior changes, not the guard.
- **`FusionFitsInBudget` has no CPU analog.** LLVM IR parameter count for many-operand fusions could blow up. **Open item:** before flipping the flag default, audit whether a parameter-count cap is needed.
- **CPU `EstimateRunTimes` uses `max(compute_time, bandwidth_time)`.** Underestimates for compute-bound duplicated producers (e.g., `log(x)` fused into 50 consumers). Acceptable for ranking; upgrade path is a self-contained change to the CPU cost model.
- **Concat `concat_minor_dim_multiplier` removed.** Hard block in `BackendCanFuse` makes it unreachable; re-add when the hard block is loosened.
- **`CpuInstructionFusion` has other behaviors not replicated here:** `CanBeOutputFused` / `CanBeOutputFusedIntoSomeOperand` (output fusion on CPU) and `IsExpensive`-based profitability filtering. The new pass replicates `IsFusible` (from `CanBeLoopFused`) + the non-scalar-constant guard (as part of CPU's `IsFusible`) + the two concat guards. **Open item: output fusion.** Today's CPU actively uses `CanBeOutputFused`; the flag-on path currently does not. Before flipping the default, benchmark against HLO that triggers output fusion today to quantify regression. If significant, either (a) teach priority fusion to produce output fusions via `ChooseKind` + a new `BackendCanFuse` branch, or (b) run `CpuInstructionFusion` in output-fusion-only mode after `CpuPriorityFusion`. Not resolved here.
- **Compile-time regression.** `CpuPriorityFusion` is expected to be 5–20× slower than `CpuInstructionFusion` in the fusion phase on a 5k-op module (~500 ms – 2 s vs ~100 ms today). Dominant source is `FusionNodeIndexingEvaluation` reconstruction; `precise_evaluation_invalidation = true` (default) keeps this closer to linear. `max_fusions_per_computation` and `thread_pool` knobs in `CpuPriorityFusionOptions` are defense-in-depth. `wall_time_budget` is **deferred** (needs base-pass deadline check). Expected end-to-end compile-time impact: +5–15%. Benchmark as part of rollout.

## Migration summary

### Phase 0 — in-place decouple (files stay at `xla/backends/gpu/transforms/`)

| Change | File |
|---|---|
| New interface (lands at final location) | `xla/hlo/transforms/fusion_cost_model.h` |
| New GPU cost model (owns GPU caches) | `xla/backends/gpu/transforms/gpu_fusion_cost_model.{h,cc}` |
| Refactor queue to use interface (in place) | `xla/backends/gpu/transforms/priority_fusion.{h,cc}` |
| New GPU subclass (next to base) | `xla/backends/gpu/transforms/gpu_priority_fusion.{h,cc}` |
| Type rename, 4 call sites | `fusion_pipeline.cc:76`, `triton_fusion_numerics_verifier.cc:~136`, `autotuner/triton.cc:~276`, `autotuner/fission_backend.cc:~127` |
| FakeCostModel-driven base-pass test | `xla/backends/gpu/transforms/priority_fusion_test.cc` |

### Phase A — file relocation (mechanical)

| Change | File |
|---|---|
| Move base pass + queue | `xla/backends/gpu/transforms/priority_fusion.{h,cc}` → `xla/hlo/transforms/priority_fusion.{h,cc}` |
| Namespace change | `xla::gpu` → `xla` |
| Promote `IsFusibleBitcast` to header | add declaration to `xla/hlo/transforms/priority_fusion.h` |
| Move tests | `priority_fusion_test.cc` to new path |
| Update include paths (no type renames) | same 4 call sites as Phase 0 |

### Phases B–D — CPU side (after relocation)

| Change | File |
|---|---|
| New CPU subclass | `xla/service/cpu/cpu_priority_fusion.{h,cc}` |
| New CPU cost model | `xla/service/cpu/cpu_fusion_cost_model.{h,cc}` |
| New tests | `cpu_priority_fusion_test.cc`, `cpu_fusion_cost_model_test.cc` |
| New flag | `xla_cpu_use_priority_fusion` in `xla/xla.proto` |
| Pipeline wiring | `xla/service/cpu/cpu_compiler.cc` (around line 1053) |

## Changelog vs v1

- v1 claimed the GPU pass was "~90% backend-neutral" — incorrect. `PriorityFusionQueue` (~850 lines) directly owned `GpuHloCostAnalysis`, `GpuPerformanceModelCache`, `GpuPerformanceModel`, `HloFusionAnalysisCache`, `MLIRContext*`, `FusionDeduplicationCache`. v2 moves the queue too and relocates those caches into `GpuFusionCostModel`.
- v1 had a 2-method interface (`Priority`, `WouldExplodeIrSize`). v2 has 5: `EstimateRunTimes` (returns `{unfused, fused}` so incremental arithmetic works), `WouldExplodeIrSize`, `OnInstructionFused`, `Invalidate`, `Revisit`.
- v1 used `set_cost_model()` post-hoc setter to work around constructor ordering. v2 has the base pass take a `std::unique_ptr<FusionCostModel>`; subclasses build it before calling the base ctor.
- v1 had `IsFusible` as a free function; CPU would attempt to fuse scatter/gather/reduce-window and crash the emitter. v2 virtualizes it on the base pass.
- v1 implemented `WouldExplodeIrSize` on CPU as `bytes > 16 KiB`, which is not what `CpuInstructionFusion` does. v2 wraps `FusionNodeIndexingEvaluation::CodeDuplicationTooHigh`.
- v1 defined `IndependentWorkItems` as `shape.dimensions(0)` (logical first dim). v2 uses `shape.dimensions(LayoutUtil::Major(shape.layout(), 0))` (physical outermost), with layout-guards and tuple/scalar handling clarified.
- v1 missed 3 GPU callers of `PriorityFusion`. v2 enumerates all 4.
- v1 `concat_minor_dim_multiplier` is removed — unreachable under the hard block.
- v1 flag-off test claimed "byte-identical HloProto". v2 calls it a smoke test for the if/else wiring.

## Changelog vs v2

v2 reviewer found 5 new blockers/majors introduced by v2's rewrite. v3 addresses each:

- **v2 dropped `InstructionFusion::ShouldFuseInPlaceOp`** from `CanFuse`'s tail — a correctness-level aliasing/in-place check. v3 retains it as the final generic step of `CanFuse` after the backend-specific hook returns Allow.
- **v2's `GpuFusionCostModel::RunInitialAnalysis` had no caller.** v3 adds `Prepare(HloComputation*)` to the `FusionCostModel` interface (6th method) and the base pass calls it before constructing the queue.
- **v2's CPU `OnInstructionFused` did `fusion_node_evaluations_.clear()`** — O(N²) compile-time. v3 uses precise `fusion_node_evaluations_.erase(consumer)`, gated by `precise_evaluation_invalidation = true` in options.
- **v2 didn't retain `operands_to_removed_consumers_runtimes_`** in the queue, which was needed for incremental priority math. v3 keeps it and documents the delta-style update path explicitly.
- **v2 hoisted `IsFusibleBitcast` to the generic `CanFuse`** before backend checks, changing GPU semantics (Triton lost first-look at bitcast-consumer cases). v3 keeps `IsFusibleBitcast` inside each backend's `BackendCanFuse` at the correct position.
- **v2's `-Inf` priority for constants** did not actually match CPU's existing "reject non-scalar constant producers" behavior. v3 puts that check in CPU's `IsFusible` override.
- **v2 called out output fusion as a non-issue.** v3 promotes it to an explicit open item with two concrete paths forward; mandatory to resolve before flipping the flag default.
- **v2 had a wrong "scan" opcode reference** (HLO has no `kScan`). v3 removes it — the actual rejected set is scatter/gather/reduce-window.
- **v2 said `tiled_run_time_data_cache_` moved to `GpuFusionCostModel` but left the queue's priority path calling `GetTiledRunTimeDataCached`** — the helper had no shared-location home. v3 folds both the cache and the helper entirely into `GpuFusionCostModel::EstimateRunTimes`.
- **v3 adds 3 compile-time knobs** to `CpuPriorityFusionOptions` (`max_fusions_per_computation`, `precise_evaluation_invalidation`, `thread_pool`) and acknowledges the expected 5–20× fusion-phase slowdown explicitly. (v3 also added `wall_time_budget`, but v4 dropped it — it needs base-pass changes out of scope.)
- **v3 pins member-ordering lifetime contract** for `CpuPriorityFusion`: `hlo_cost_analysis_` is captured by pointer inside `cost_model_` during base-class construction; the member declaration order keeps the analysis alive for the cost model's lifetime. Documented with an in-header comment to prevent future reorderings from breaking it.

## Changelog vs v3

v3 reviewer found 4 plumbing blockers in the CPU path. v4 fixes:

- CPU `Prepare` override added: `CpuFusionCostModel::Prepare(computation)` runs `computation->Accept(hlo_cost_analysis_)`. Without this, every `Kflops()` call returned 0.
- Bitcast-consumer forbid added to step 1 of CPU `BackendCanFuse`. Spec already listed it; the plan's code block was missing it.
- `CpuFusionCostModel` ctor now takes `bool precise_invalidation` so `OnInstructionFused` uses `erase(consumer)` rather than `clear()` — v3 had declared the option but left the field uninitialized.
- `CpuPriorityFusion` now passes `options.thread_pool` to the base ctor instead of hardcoding `nullptr`.
- `wall_time_budget` dropped from v4 — requires base-pass changes out of scope. Marked as follow-up in Open Items.
- Incremental priority math now shown field-by-field (`unfused` and `fused` subtract independently) in plan Task A2 Step 5, rather than prose only.
- GPU `IsFusible` documented as explicitly inherited from base default.
- Lifetime-ordering comment extended to cover destruction order and prohibit `~CpuFusionCostModel` dereferencing `hlo_cost_analysis_`.

## Changelog vs v4

v4 reviewer confirmed all v3 plumbing fixes landed cleanly, but found 2 compile-level blockers and 2 prose inconsistencies. v5 fixes:

- **Ctor arity shear.** Widening `CpuFusionCostModel` to take `bool precise_invalidation` broke Phase B tests. v5 adds `= true` default so tests keep compiling without updates.
- **`IsFusibleBitcast` file-local.** Pre-existing; v4's fix to #11 surfaced it. v5 promotes its declaration to `xla/hlo/transforms/priority_fusion.h` so both GPU and CPU subclass `.cc` files can call it.
- Spec §CPU cost model dependency list updated to `HloCostAnalysis*` (non-const).
- Spec Risks section no longer claims `wall_time_budget` is in the struct.
- Plan's incremental `CalculateProducerPriority` code now uses `reverse_map_.at(producer)->first.first` instead of a placeholder comment.
- GPU `BackendCanFuse` step numbering fixed (was 1, 2, 3, 3, 4, 5).
- Task A2 Step 8 pins down that `OnInstructionFused` is called from inside `Fuse()` at the tail.

## Changelog vs v5

- **Phasing restructured.** v5 had a single Phase A that combined the risky refactor (extracting `PriorityFusionQueue`'s GPU state, adding virtuals, creating subclass) with the risky file move, and forced a multi-task atomic commit. v6 splits into:
  - **Phase 0 (in-place decouple):** all refactoring happens at `xla/backends/gpu/transforms/`. Each task commits standalone. Full GPU test suite passes after every task.
  - **Phase A (file relocation):** mechanical `git mv` + namespace/guard/BUILD changes. Zero logic change. Because files are already decoupled, the move is low-risk.
- **No more atomic multi-task commits.** Every task in Phase 0 commits on its own and builds/tests cleanly.
- **`IsFusibleBitcast` declaration lands in Phase 0's in-place header** (`xla/backends/gpu/transforms/priority_fusion.h`). Phase A carries it along during the move; nothing changes in the subclass `.cc` files other than the include path.
- **Updated Migration summary table** to reflect the two-stage migration.
