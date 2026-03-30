# Unified XLA Fusion Pipeline

**Authors:** seantalts@
**Status:** Draft
**Last Updated:** 2026-03-30

## Objective

Refactor XLA's fusion infrastructure so that CPU and GPU share a single
`PriorityFusion` algorithm parameterized by pluggable cost models and library
matchers. This eliminates ~700 lines of CPU-specific fusion code, gives CPU
the benefit of GPU's battle-tested priority-queue fusion algorithm, and creates
a clean extension point for new backends and accelerator libraries.

## Background

### Current GPU fusion

GPU uses `PriorityFusion` (`xla/backends/gpu/transforms/priority_fusion.cc`), a
priority-queue-based greedy algorithm:

1. For each producer, estimate `priority = time_unfused - time_fused` using
   `GpuPerformanceModel`.
2. Push producers with positive priority into a queue.
3. Pop highest-priority producer, fuse into consumers.
4. Update priorities of affected instructions. Repeat until queue is empty.

This is followed by `MultiOutputFusion` for sibling fusion, then cleanup passes.

The cost model (`GpuPerformanceModel`) estimates kernel launch overhead, memory
bandwidth, compute throughput, and L1/L2 cache effects. Fusion legality checks
(`CanFuse`) include Triton-specific tiling analysis, ptxas parameter budget,
shared memory limits, and emitter compatibility.

### Current CPU fusion

CPU uses three simpler passes:

1. **`CpuInstructionFusion`** — extends `InstructionFusion` base class.
   Single-pass reverse-post-order traversal. Fusion decisions use a boolean
   `IsExpensive()` heuristic plus hard-coded limits (max 5 reductions, max 8
   concat args, 16KB dot threshold).

2. **`FusionWrapper`** — wraps individual HLO ops into fusion nodes so the MLIR
   emitter can handle them.

3. **`CpuMultiOutputFusion`** — extends `MultiOutputFusion` base class. Profit
   metric is bytes of shared operands. Max 20 operands per fusion.

### Problem

The CPU fusion pipeline is simplistic and ad-hoc. Adding new libraries
(YNNPACK, OneDNN) requires modifying CPU-specific fusion code. The GPU pipeline
is sophisticated but hardcodes GPU-specific concerns (Triton, cuBLAS, ptxas
limits) directly into the algorithm. Neither is extensible.

## Overview

```
                        +---------------------+
                        |  PriorityFusion     |
                        |  (generic algorithm) |
                        +----------+----------+
                                   |
                    +--------------+--------------+
                    |                             |
          +---------v---------+        +----------v--------+
          | FusionCostModel   |        | FusionLegality    |
          | (interface)       |        | Checker (interface)|
          +----+---------+----+        +----+---------+----+
               |         |                  |         |
        +------v-+  +----v-----+     +------v-+  +---v-------+
        |GpuCost |  |CpuCost   |     |GpuLegal|  |CpuLegal   |
        |Model   |  |Model     |     |ity     |  |ity        |
        +--------+  +----------+     +--------+  +-----------+
                                          |
                                   +------v-------+
                                   |LibraryMatcher|
                                   | (interface)  |
                                   +--+--+--+--+--+
                                      |  |  |  |
                            +---------+  |  |  +----------+
                            |            |  |             |
                      +-----v---+ +-----v--v-+ +--------v---+
                      |Triton   | |CuBlas/   | |YNNPACK     |
                      |Matcher  | |CuDnn     | |Matcher     |
                      +---------+ |Matcher   | +------------+
                                  +----------+
```

The key interfaces:

- **`FusionCostModel`** — returns `RunTimes{time_unfused, time_fused}` for a
  proposed fusion. Drives priority calculation.
- **`FusionLegalityChecker`** — returns `FusionDecision` for whether a
  producer-consumer pair can legally be fused. Composes generic checks
  (cycle-freedom, in-place ops) with backend-specific checks (parameter budgets,
  emitter compatibility).
- **`LibraryMatcher`** — determines whether an instruction (or
  producer-consumer pair) should be handled by an external library rather than
  fused. Multiple matchers can be registered per backend.

## Detailed Design

### 1. `RunTimes` and `EstimateRunTimeData` (move to shared location)

These structures are currently in `xla/service/gpu/model/gpu_performance_model_base.h`
but contain nothing GPU-specific:

```cpp
// xla/service/fusion_cost_model.h (new file)

struct EstimateRunTimeData {
  int64_t flops;
  int64_t bytes_read;
  int64_t bytes_written;
  absl::Duration read_time;
  absl::Duration write_time;
  absl::Duration compute_time;
  absl::Duration exec_time;

  static EstimateRunTimeData Zero();
  static EstimateRunTimeData Infinite();
  bool IsInfinite() const;
};

struct RunTimes {
  absl::Duration time_unfused;
  absl::Duration time_fused;
};
```

### 2. `FusionCostModel` interface

```cpp
// xla/service/fusion_cost_model.h

class FusionCostModel {
 public:
  virtual ~FusionCostModel() = default;

  // Estimate the runtime benefit of fusing producer into its consumers.
  // Returns time_unfused (sum of separate kernels/calls) and time_fused
  // (single fused kernel/call).
  virtual RunTimes EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<const HloInstruction* const> fused_consumers) = 0;

  // Estimate runtime for a multi-output fusion of producer and consumer.
  virtual RunTimes EstimateRunTimesForMultiOutputFusion(
      const HloInstruction* producer,
      const HloInstruction* consumer) = 0;

  // Per-instruction runtime estimate (used for caching and incremental
  // priority updates).
  virtual EstimateRunTimeData EstimateRunTimeForInstruction(
      const HloInstruction* instr) = 0;

  // Called when an instruction is modified (fusion, deletion) so the
  // cost model can invalidate caches.
  virtual void InvalidateCache(const HloInstruction* instr) = 0;
};
```

**GPU implementation** (`GpuFusionCostModel`): thin wrapper around existing
`GpuPerformanceModel` + `GpuHloCostAnalysis`. No behavior change.

**CPU implementation** (`CpuFusionCostModel`): initial version translates
current heuristics into the cost model interface:

| Current CPU heuristic | Cost model translation |
|---|---|
| `IsExpensive(op) = true` | `time_fused = Infinite` (never fuse as producer) |
| Library-backed dot/conv | `time_fused >> time_unfused` (library call is faster) |
| Elementwise chain | `time_fused = time_unfused - memory_saved` (fusion wins) |
| Max 5 reductions | After 5th reduction, `time_fused = Infinite` |

Over time, the CPU cost model evolves to use actual bandwidth/FLOPS estimates
based on CPU microarchitecture (cache sizes, SIMD width, core count).

### 3. `FusionLegalityChecker` interface

```cpp
// xla/service/fusion_legality_checker.h

class FusionLegalityChecker {
 public:
  virtual ~FusionLegalityChecker() = default;

  // Can producer be fused into consumer?
  virtual FusionDecision CanFuse(const HloInstruction* producer,
                                 const HloInstruction* consumer) = 0;

  // Can producer be fused as a multi-output fusion with consumer?
  virtual FusionDecision CanFuseMultiOutput(
      const HloInstruction* producer,
      const HloInstruction* consumer) = 0;

  // Choose fusion kind (kLoop, kInput, kOutput, kCustom).
  virtual HloInstruction::FusionKind ChooseKind(
      const HloInstruction* producer,
      const HloInstruction* consumer) = 0;

  // Has the fused result grown too large to emit?
  virtual bool FusedResultTooLarge(const HloInstruction* producer,
                                   const HloInstruction* consumer) = 0;

  // Called when an instruction is modified.
  virtual void InvalidateCache(const HloInstruction* instr) = 0;
};
```

The implementation composes **generic checks** (shared by all backends) with
**backend-specific checks**:

**Generic checks** (extracted from current GPU code into a shared helper):
- Producer is not the root instruction
- No cycles created by fusion (reachability)
- `ShouldFuseInPlaceOp()` constraints (from `InstructionFusion` base)
- Single-user bitcast consumer skip

**GPU-specific checks** (in `GpuFusionLegalityChecker`):
- `FusionFitsInBudget()` — shared memory, parameter limit (96), unnested
  reductions
- Emitter compatibility — reduction-into-reduction prevention, emitter fusion
  kind switching
- `ProducerConsumerMergedTooLarge()` — IR size estimation
- Triton constraints (delegated to `TritonMatcher`)

**CPU-specific checks** (in `CpuFusionLegalityChecker`):
- Max operand/concat/reduction limits (from current `ShouldFuse`)
- Code duplication threshold (`FusionNodeIndexingEvaluation`)
- Dot fusion restrictions (matrix-vector only, <16KB)

### 4. `LibraryMatcher` interface

```cpp
// xla/service/library_matcher.h

class LibraryMatcher {
 public:
  virtual ~LibraryMatcher() = default;

  // Name of the library (for logging/debugging).
  virtual absl::string_view name() const = 0;

  // Should this instruction be handled by the library instead of fused?
  // Returns Allow if the library does NOT claim the instruction (fusion
  // may proceed). Returns Forbid if the library claims it.
  virtual FusionDecision ShouldFuseInsteadOfLibrary(
      const HloInstruction* producer,
      const HloInstruction* consumer) = 0;

  // Can producer be fused into consumer when one or both are library
  // fusions? (e.g., Triton can absorb elementwise producers.)
  virtual FusionDecision CanFuseWithLibraryOp(
      const HloInstruction* producer,
      const HloInstruction* consumer) = 0;
};
```

**Implementations:**

| Library | `ShouldFuseInsteadOfLibrary` | `CanFuseWithLibraryOp` |
|---|---|---|
| `TritonMatcher` | Checks `IsTritonSupported`, runs tiling analysis via `GpuIndexingPerformanceModel::TryFindBestTilingForFusion` | Absorbs elementwise producers into Triton fusions |
| `CuBlasGemmMatcher` | Matches GEMM patterns for cuBLAS dispatch | Cannot fuse into cuBLAS calls |
| `CuDnnMatcher` | Matches convolution/attention patterns | Cannot fuse into cuDNN calls |
| `YnnpackMatcher` | Checks `IsDotSupportedByYnn`, `IsConvolutionOpSupportedByYnn`, `IsReduceLikeOpOffloadedToYnn` | Cannot fuse into YNNPACK calls |
| `OneDnnMatcher` | Checks `OneDnnContractionRewriter::ShouldRewriteInstr` | Cannot fuse into OneDNN calls |

`FusionLegalityChecker` consults all registered `LibraryMatcher`s as part of
its `CanFuse` check. If any matcher claims an instruction, fusion is forbidden
(the library call is expected to be faster).

### 5. Refactored `PriorityFusion`

```cpp
// xla/service/priority_fusion.h (moved from xla/backends/gpu/transforms/)

class PriorityFusion : public HloModulePass {
 public:
  PriorityFusion(
      tsl::thread::ThreadPool* thread_pool,
      FusionCostModel* cost_model,
      FusionLegalityChecker* legality_checker,
      absl::Span<LibraryMatcher* const> library_matchers,
      const AliasInfo* alias_info);

  absl::string_view name() const override { return "priority-fusion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;
};
```

The algorithm itself (priority queue, incremental updates, caching) does not
change. What changes is that every call to `GpuPerformanceModel`,
`GpuHloCostAnalysis`, `CanFuseTriton`, `FusionFitsInBudget`, etc. is replaced
with calls through the interfaces above.

### 6. Backend wiring

**GPU** (`gpu_compiler.cc::RunFusionPasses`):

```cpp
GpuFusionCostModel cost_model(device_info, cost_analysis_options,
                               &fusion_analysis_cache, mlir_context);
GpuFusionLegalityChecker legality(device_info, &cost_analysis,
                                   &fusion_analysis_cache);
TritonMatcher triton(device_info, mlir_context);
CuBlasGemmMatcher cublas(device_info);

std::vector<LibraryMatcher*> matchers = {&triton, &cublas};
pipeline.AddPass<PriorityFusion>(thread_pool, &cost_model, &legality,
                                  matchers, alias_info);
pipeline.AddPass<gpu::MultiOutputFusion>(...);
```

**CPU** (`cpu_compiler.cc::RunHloPassesAfterLayoutAssn`):

```cpp
CpuFusionCostModel cost_model(target_machine_features);
CpuFusionLegalityChecker legality(target_machine_features, alias_info);

std::vector<LibraryMatcher*> matchers;
YnnpackMatcher ynnpack;
OneDnnMatcher onednn;
if (use_ynnpack) matchers.push_back(&ynnpack);
if (use_onednn) matchers.push_back(&onednn);

pipeline.AddPass<PriorityFusion>(thread_pool, &cost_model, &legality,
                                  matchers, &alias_info);
pipeline.AddPass<CpuMultiOutputFusion>(&alias_info);  // or unified MOF
```

This replaces `CpuInstructionFusion` entirely. `FusionWrapper` remains (both
backends need it for their respective emitters).

## Migration Plan

### Phase 1: Extract shared data structures (1-2 weeks)

- Move `RunTimes`, `EstimateRunTimeData` to `xla/service/fusion_cost_model.h`.
- Define `FusionCostModel`, `FusionLegalityChecker`, `LibraryMatcher`
  interfaces.
- GPU typedefs or trivial wrappers to maintain source compatibility.
- **No behavior change. All existing tests pass.**

### Phase 2: Implement `GpuFusionCostModel` (2-3 weeks)

- Wrap `GpuPerformanceModel` in the `FusionCostModel` interface.
- Wrap GPU legality checks in `GpuFusionLegalityChecker`.
- Extract `TritonMatcher` from inline code in `PriorityFusionQueue::CanFuse`.
- Refactor `PriorityFusion` to accept the interfaces instead of concrete types.
- **No behavior change. All existing GPU tests pass.**

### Phase 3: Implement `CpuFusionCostModel` (2-3 weeks)

- Translate `CpuInstructionFusion::IsExpensive` + `ShouldFuse` heuristics into
  `CpuFusionCostModel::EstimateRunTimes`.
- Implement `CpuFusionLegalityChecker` with current CPU limits.
- Implement `YnnpackMatcher` and `OneDnnMatcher`.
- **CPU fusion decisions should be equivalent to today's.** Validate with
  existing CPU tests.

### Phase 4: Wire CPU to `PriorityFusion` (1-2 weeks)

- Replace `CpuInstructionFusion` with `PriorityFusion` + CPU cost model.
- Run full CPU test suite. Investigate any diffs (expected: minor ordering
  changes that don't affect correctness).
- Delete `CpuInstructionFusion`.

### Phase 5: Iterate on `CpuFusionCostModel` (ongoing)

- Replace heuristics with actual performance estimates (memory bandwidth,
  FLOPS, cache effects) based on CPU target machine features.
- Benchmark and tune. The priority-fusion algorithm will automatically make
  better decisions as the cost model improves.

### Phase 6: Unify `MultiOutputFusion` (stretch)

- The same interface pattern applies. GPU's `MultiOutputFusion` already uses
  `GpuPerformanceModel::EstimateRunTimesForMultiOutputFusion`. Parameterize
  it by `FusionCostModel` and reuse for CPU.

## Risks and Mitigations

### Risk: CPU fusion regressions

**Mitigation:** Phase 3 explicitly translates current heuristics 1:1. We
validate by diffing the fusion decisions (using `--xla_dump_to` and comparing
fused HLO) on a benchmark suite before switching. The priority-queue algorithm
with equivalent cost estimates should produce equivalent-or-better results
because it considers global benefit ordering rather than fixed traversal order.

### Risk: GPU performance regressions from refactoring

**Mitigation:** Phase 2 is a pure refactor behind interfaces. We keep the GPU
cost model implementation identical and gate the change behind an
A/B-testable flag. GPU fusion process dumps (`FusionProcessDumpProto`) provide
deterministic diffing of fusion decisions.

### Risk: Interface overhead

**Mitigation:** Virtual dispatch cost is negligible compared to the cost model
computations themselves (which involve FLOPS estimation, tiling analysis, etc.).
The hot path (priority queue operations) remains non-virtual.

### Risk: Triton/library matcher complexity

**Mitigation:** `TritonMatcher` is the most complex library matcher because it
performs tiling analysis during `CanFuse`. We keep this complexity inside the
matcher rather than trying to simplify it. The interface just gives it a clean
boundary.

## Alternatives Considered

### Alternative 1: Keep separate fusion passes, share only utilities

Share helper functions (e.g., `CreateSimplificationPipeline`) but keep
`CpuInstructionFusion` and `PriorityFusion` as separate passes.

**Rejected:** This perpetuates the maintenance burden. Every new feature
(library integration, cost model improvement) must be implemented twice. The
CPU pass would remain stuck with its single-pass heuristic algorithm.

### Alternative 2: Make CPU use GPU's `PriorityFusion` directly with GPU cost model

Pass a "fake GPU" device description to `GpuPerformanceModel` with CPU-like
parameters (memory bandwidth, no kernel launch overhead, etc.).

**Rejected:** The GPU cost model makes fundamentally GPU-specific assumptions
(kernel launch overhead, warp scheduling, shared memory tiling). Tweaking
parameters wouldn't fix the structural mismatch. A proper CPU cost model
accounts for CPU-specific effects (cache hierarchy, SIMD vectorization,
function call overhead vs kernel launch overhead).

### Alternative 3: Template-based approach instead of virtual interfaces

Use templates (`PriorityFusion<GpuCostModel>`) instead of virtual dispatch.

**Rejected:** Templates prevent separate compilation and make the code harder
to test. The virtual dispatch overhead is negligible in this context. Interfaces
also enable runtime composition (e.g., registering library matchers via a
plugin mechanism).

## Testing Strategy

1. **Unit tests for each interface implementation:** `GpuFusionCostModel`,
   `CpuFusionCostModel`, each `LibraryMatcher`.
2. **Fusion decision diffing:** Compare `FusionProcessDump` protos before and
   after refactoring to verify no GPU behavior change.
3. **HLO snapshot tests:** For CPU, compare fused HLO output on a set of
   representative models before and after the switch.
4. **Performance benchmarks:** Run standard benchmarks on both GPU and CPU to
   detect regressions. Gate rollout behind feature flags.
5. **Existing test suites:** All existing `cpu_instruction_fusion_test`,
   `priority_fusion_test`, `multi_output_fusion_test` must continue to pass.
