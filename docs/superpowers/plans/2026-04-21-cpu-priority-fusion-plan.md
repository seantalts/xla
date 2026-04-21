# CPU Priority Fusion Implementation Plan (v6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move XLA's GPU priority fusion pass (including its nested `PriorityFusionQueue`) to a shared location behind a 6-method `FusionCostModel` interface, relocate GPU-specific state into `GpuFusionCostModel`, and add a CPU subclass with a kflops/work-items guard.

**Architecture:** Shared base pass + queue at `xla/hlo/transforms/`, with zero GPU types in the shared code. Thin GPU subclass with same constructor signature as today's `PriorityFusion`. New CPU subclass + cost model. Guard (`kflops > X AND work_items < 32`) in `CpuPriorityFusion::BackendCanFuse`. Complex ops get inflated per-element kflops.

**Tech Stack:** C++17 (XLA convention), absl, gtest, Bazel, `HloCostAnalysis`, `GpuPerformanceModel` (GPU side).

**Spec:** `docs/superpowers/specs/2026-04-21-cpu-priority-fusion-design.md` (v6)

**Branch:** `claude/gpu-priority-fusion-xla-kBHi3`

**Supersedes:** v1, v2, v3, v4, v5. v6 restructures the refactor into two stages: Phase 0 decouples GPU state from the shared code **in place** (no file moves), and Phase A relocates the now-decoupled files mechanically. This eliminates intermediate states where generic code could depend on GPU code, and makes each Phase 0 step a standalone commit (no more atomic multi-task commits).

---

## Pre-flight: verify the starting state

- [ ] **Step 0.1: Confirm branch**

Run: `git -C /home/user/xla branch --show-current`
Expected: `claude/gpu-priority-fusion-xla-kBHi3`.

- [ ] **Step 0.2: Enumerate all GPU call sites**

Run:
```bash
grep -rn "gpu::PriorityFusion\b\|backends/gpu/transforms/priority_fusion\.h" \
  --include='*.cc' --include='*.h' --include='BUILD' /home/user/xla/xla
```

Expected 4 call sites plus BUILDs: `fusion_pipeline.cc:76`, `triton_fusion_numerics_verifier.cc:~136`, `autotuner/triton.cc:~276`, `autotuner/fission_backend.cc:~127`. If more appear, add them to Task A4 Step 1.

- [ ] **Step 0.3: Confirm C++ standard**

Run: `grep -r "cxxopt.*std=c++\|CXXFLAGS.*c++" /home/user/xla/.bazelrc 2>/dev/null | head -3`
Expected: `c++17`. If `c++20`, designated-initializer syntax (`Options{.shape_size=...}`) is plain. This plan assumes **C++17**; designated initializers are a Clang extension that works in XLA's build but are replaced with positional constructors or a helper in case of doubt.

---

## File Structure

### New files

| Path | Purpose |
|---|---|
| `xla/hlo/transforms/BUILD` | Bazel build file |
| `xla/hlo/transforms/fusion_cost_model.h` | 5-method `FusionCostModel` interface |
| `xla/hlo/transforms/priority_fusion.h` | Shared pass header (moved) |
| `xla/hlo/transforms/priority_fusion.cc` | Shared pass + queue (moved) |
| `xla/hlo/transforms/priority_fusion_test.cc` | Base pass tests + FakeCostModel |
| `xla/backends/gpu/transforms/gpu_priority_fusion.{h,cc}` | Thin GPU subclass |
| `xla/backends/gpu/transforms/gpu_fusion_cost_model.{h,cc}` | Owns all GPU cost-model state |
| `xla/service/cpu/cpu_fusion_cost_model.{h,cc}` | CPU cost model |
| `xla/service/cpu/cpu_fusion_cost_model_test.cc` | Unit tests |
| `xla/service/cpu/cpu_priority_fusion.{h,cc}` | CPU subclass + guard |
| `xla/service/cpu/cpu_priority_fusion_test.cc` | CPU tests |

### Modified files

| Path | Purpose |
|---|---|
| `xla/xla.proto` | Add `xla_cpu_use_priority_fusion` flag (field 503) |
| `xla/backends/gpu/transforms/BUILD` | Remove old, add new targets |
| `xla/service/gpu/fusion_pipeline.cc` (~76) | Type rename |
| `xla/backends/gpu/transforms/triton_fusion_numerics_verifier.cc` (~136) | Type rename |
| `xla/backends/gpu/autotuner/triton.cc` (~276) | Type rename |
| `xla/backends/gpu/autotuner/fission_backend.cc` (~127) | Type rename |
| Respective `BUILD` files | Dep updates |
| `xla/service/cpu/cpu_compiler.cc` (~1053) | Flag branch |
| `xla/service/cpu/BUILD` | New CPU target deps |

### Deleted files (via `git mv`)

- `xla/backends/gpu/transforms/priority_fusion.h`
- `xla/backends/gpu/transforms/priority_fusion.cc`
- `xla/backends/gpu/transforms/priority_fusion_test.cc`

---

## Phase 0 — In-place decouple (zero file moves, zero GPU behavior change)

**Goal of Phase 0:** split `PriorityFusionQueue`'s GPU state into a `GpuFusionCostModel`, introduce the `FusionCostModel` interface, hoist the base-class virtuals, and create a `GpuPriorityFusion` subclass — all **without moving any files**. Base pass + queue stay at `xla/backends/gpu/transforms/priority_fusion.{h,cc}` with `namespace xla::gpu`. Phase A later relocates the decoupled files mechanically.

**Phase 0 commit strategy.** Each task commits standalone. After every task, the full GPU test suite (`bazel test //xla/backends/gpu/transforms:priority_fusion_test`) must pass. No atomic multi-task commits needed — in-place refactoring means every intermediate state is buildable.

### Task 0.1: Add the `FusionCostModel` interface

**Files:**
- Modify: `xla/hlo/transforms/BUILD` (add new `fusion_cost_model` target; file already exists)
- Create: `xla/hlo/transforms/fusion_cost_model.h`

- [ ] **Step 1: Add the BUILD target**

Append to the existing `xla/hlo/transforms/BUILD`:

```python
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

cc_library(
    name = "fusion_cost_model",
    hdrs = ["fusion_cost_model.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)
```

(The `load(...)` and `package(...)` stanzas are already present in the existing BUILD file — do not duplicate them.)

- [ ] **Step 2: Create the header**

```cpp
/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_HLO_TRANSFORMS_FUSION_COST_MODEL_H_
#define XLA_HLO_TRANSFORMS_FUSION_COST_MODEL_H_

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// Abstract cost model consulted by `PriorityFusion` and its queue. The
// base pass calls into implementations from five places: initial
// per-computation setup (Prepare), priority computation
// (EstimateRunTimes), IR-size budget (WouldExplodeIrSize),
// cache invalidation when instructions are removed (Invalidate,
// OnInstructionFused), and re-costing after an instruction's inputs
// change (Revisit). Backends that don't need caching default the
// last three to no-ops.
class FusionCostModel {
 public:
  struct RunTimes {
    // `producer` runs once plus each consumer runs once.
    absl::Duration unfused;
    // Each consumer runs once with the producer inlined into it.
    absl::Duration fused;
  };

  virtual ~FusionCostModel() = default;

  // Called once per computation before any EstimateRunTimes calls. GPU
  // uses this to run its initial cost-analysis Accept() and populate
  // internal caches. Default: no-op.
  virtual absl::Status Prepare(HloComputation* computation) {
    return absl::OkStatus();
  }

  // Returns the estimated runtime of producing `producer` then running
  // each of `consumers`, both unfused and fused. The base pass computes
  // `priority = unfused - fused` for queue ordering.
  virtual RunTimes EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<HloInstruction* const> consumers) = 0;

  // True if fusing `producer` into `consumer` would produce code that
  // exceeds the backend's code-size budget. GPU wraps
  // GpuHloCostAnalysis::ProducerConsumerMergedTooLarge. CPU wraps
  // FusionNodeIndexingEvaluation::CodeDuplicationTooHigh.
  virtual bool WouldExplodeIrSize(const HloInstruction* producer,
                                  const HloInstruction* consumer) = 0;

  // Called once per successful fusion. Default: no-op.
  virtual void OnInstructionFused(HloInstruction* producer,
                                  HloInstruction* consumer,
                                  HloInstruction* fusion) {}

  // Called when an instruction's cached cost must be dropped (user set
  // changed, about to be removed, etc.). Default: no-op.
  virtual void Invalidate(const HloInstruction* instruction) {}

  // Called when an instruction's inputs changed and its underlying cost
  // analysis must be recomputed. Default: ok, no-op.
  virtual absl::Status Revisit(const HloInstruction* instruction) {
    return absl::OkStatus();
  }
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_FUSION_COST_MODEL_H_
```

- [ ] **Step 3: Build**

Run: `bazel build //xla/hlo/transforms:fusion_cost_model`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add xla/hlo/transforms/BUILD xla/hlo/transforms/fusion_cost_model.h
git commit -m "Add FusionCostModel interface for shared priority fusion"
```

---

### Task 0.2: Refactor `PriorityFusionQueue` to use `FusionCostModel` (in place)

Files stay at `xla/backends/gpu/transforms/priority_fusion.{h,cc}` with `namespace xla::gpu`. The goal is that the queue class no longer holds any GPU types as direct members — it holds a `FusionCostModel*` and a `backend_can_fuse` functor. Phase A later relocates these now-decoupled files.

**Files:**
- Modify: `xla/backends/gpu/transforms/priority_fusion.h` (base pass gains virtuals + unique_ptr<FusionCostModel>)
- Modify: `xla/backends/gpu/transforms/priority_fusion.cc` (queue loses GPU members, uses interface)
- Modify: `xla/backends/gpu/transforms/BUILD` (add dep on `//xla/hlo/transforms:fusion_cost_model`)

- [ ] **Step 1: (skipped in Phase 0 — no file moves)**

Phase 0 does not `git mv` anything. Files stay where they are. The namespace stays `xla::gpu`. Phase A handles the relocation later as a pure mechanical step.

- [ ] **Step 2: Rewrite the header**

Open the old header. Replace its entire content with:

```cpp
/* Copyright 2017 The OpenXLA Authors. [standard header, keep existing year] */

// Include guard and namespace stay as-is for Phase 0. Phase A will
// rename XLA_BACKENDS_GPU_TRANSFORMS_PRIORITY_FUSION_H_ to
// XLA_HLO_TRANSFORMS_PRIORITY_FUSION_H_ and `namespace xla::gpu`
// to `namespace xla` as part of the file relocation.
#ifndef XLA_BACKENDS_GPU_TRANSFORMS_PRIORITY_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_PRIORITY_FUSION_H_

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/transforms/fusion_cost_model.h"  // interface at final location
#include "xla/service/gpu/fusion_process_dump.pb.h"
#include "xla/service/instruction_fusion.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

class PriorityFusion : public HloModulePass {
 public:
  PriorityFusion(tsl::thread::ThreadPool* thread_pool,
                 const AliasInfo* alias_info,
                 std::unique_ptr<FusionCostModel> cost_model);

  absl::string_view name() const override { return "priority-fusion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 protected:
  // Hooks for backend subclasses.
  virtual FusionDecision BackendCanFuse(HloInstruction* producer,
                                        HloInstruction* consumer) {
    return FusionDecision::Allow();
  }
  virtual HloInstruction::FusionKind ChooseKind(const HloInstruction* producer,
                                                 const HloInstruction* consumer) {
    return HloInstruction::FusionKind::kLoop;
  }
  // Default: the GPU allowlist (elementwise, broadcast, reduce, gather,
  // scatter, reduce-window, ...). CPU overrides to drop scatter,
  // reduce-window, gather, and non-scalar constant producers.
  virtual bool IsFusible(const HloInstruction& instruction);

  FusionCostModel* cost_model() { return cost_model_.get(); }

  HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer,
                       bool use_multi_output_fusion = false);

 private:
  bool ConsumeFuel(HloInstruction* producer, HloInstruction* consumer);
  FusionDecision CanFuseConstant(const HloInstruction* constant,
                                 const HloInstruction* user);

  tsl::thread::ThreadPool* thread_pool_;
  const AliasInfo* alias_info_;
  std::unique_ptr<FusionCostModel> cost_model_;
  std::unique_ptr<FusionProcessDumpProto> fusion_process_dump_;
};

// Free helper used by backend `BackendCanFuse` implementations. Today
// this lives as a file-local static in priority_fusion.cc; it needs
// a header declaration now because both GPU subclass (GpuPriorityFusion,
// Task 0.3) and later CPU subclass call it from different TUs.
bool IsFusibleBitcast(const HloInstruction& instr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_PRIORITY_FUSION_H_
```

Note: if `fusion_process_dump.proto` is under `xla/service/gpu/`, leave the include path as-is (`"xla/service/gpu/fusion_process_dump.pb.h"`) — the proto itself doesn't need to move. The base pass depends on it for debug logging, which is backend-neutral.

- [ ] **Step 3: Edit `priority_fusion.cc` — includes and ownership (namespace unchanged in Phase 0)**

At the top of the file:
- **Keep** `namespace xla { namespace gpu {` as-is. Phase A changes this later.
- **Keep** the existing `#include "xla/backends/gpu/transforms/priority_fusion.h"` (own header, still at current path).
- Remove includes of `GpuHloCostAnalysis`, `GpuPerformanceModel`, `GpuPerformanceModelCache`, `HloFusionAnalysisCache`, `FusionDeduplicationCache`, `CanEmitInputFusedScatter`, `FusionFitsInBudget`, `IsGenericTritonFusion`, `IsTritonSupportedInstruction`, `BlockLevelParameters`, `TiledRunTimeData`, `mlir/IR/MLIRContext.h`, `se::DeviceDescription`. These move to `gpu_fusion_cost_model.h` / `gpu_priority_fusion.cc` (Task 0.3).
- Remove the `using ::xla::gpu::...` lines that reference those types.
- Add `#include "xla/hlo/transforms/fusion_cost_model.h"`.
- Add `#include "absl/functional/any_invocable.h"`.

- [ ] **Step 4: Rewrite `PriorityFusionQueue` constructor and fields**

`PriorityFusionQueue` currently (around `priority_fusion.cc:158–205`) takes GPU types. Replace with:

```cpp
class PriorityFusionQueue {
 public:
  using BackendCanFuseFn =
      absl::AnyInvocable<FusionDecision(HloInstruction*, HloInstruction*)>;

  PriorityFusionQueue(HloComputation* computation,
                      FusionCostModel* cost_model,
                      BackendCanFuseFn backend_can_fuse,
                      const AliasInfo* alias_info,
                      tsl::thread::ThreadPool* thread_pool,
                      FusionProcessDumpProto* fusion_process_dump,
                      absl::AnyInvocable<bool(const HloInstruction&)> is_fusible);
  // ...
 private:
  FusionCostModel* cost_model_;                // not owned
  BackendCanFuseFn backend_can_fuse_;
  absl::AnyInvocable<bool(const HloInstruction&)> is_fusible_;
  HloComputation* computation_;
  tsl::thread::ThreadPool* thread_pool_;
  FusionProcessDumpProto* fusion_process_dump_;
  const AliasInfo* alias_info_;

  // Generic caches kept inside the queue.
  absl::flat_hash_map<HloInstruction*,
                      absl::flat_hash_map<const HloInstruction*, FusionDecision>>
      can_fuse_cache_;
  absl::Mutex can_fuse_cache_mutex_;

  // Priority queue and reverse map — unchanged from v1 of the file.
  std::map<std::pair<absl::Duration, int64_t>, HloInstruction*> producer_priority_queue_;
  absl::flat_hash_map<HloInstruction*,
                      std::map<std::pair<absl::Duration, int64_t>, HloInstruction*>::iterator>
      reverse_map_;
  absl::flat_hash_map<HloInstruction*,
                      absl::flat_hash_set<HloInstruction*>>
      operands_to_new_consumers_;
  // ...
};
```

**Remove from the queue class:** `cost_analysis_` (was `GpuHloCostAnalysis`), `gpu_performance_model_cache_`, `gpu_performance_model_`, `fusion_analysis_cache_`, `mlir_context_`, `fusion_deduplication_cache_`, `tiled_run_time_data_cache_`, `gpu_indexing_performance_model_`. These are all now inside `GpuFusionCostModel`.

**Explicitly KEEP in the queue:** `operands_to_removed_consumers_runtimes_` — its payload is `FusionCostModel::RunTimes` (generic), and it is required for the incremental priority-update math. The map survives the move unchanged; only its type changes from `GpuPerformanceModel::RunTimes` to `FusionCostModel::RunTimes` (same fields, new namespace). Without it, `CalculateProducerPriority` would need to pass the full `producer->users()` span to `EstimateRunTimes` every time the consumer set changes, which is O(N²) compile time.

- [ ] **Step 5: Rewrite every queue method that touched removed members**

Search for each removed member and replace the call with a `cost_model_` equivalent:

| Old queue code | Replace with |
|---|---|
| `gpu_performance_model_.EstimateRunTimes(producer, ...)` | `cost_model_->EstimateRunTimes(producer, consumers)` |
| `cost_analysis_.ProducerConsumerMergedTooLarge(p, c)` | `cost_model_->WouldExplodeIrSize(&p, &c)` |
| `gpu_performance_model_cache_.Invalidate(*instr)` | `cost_model_->Invalidate(instr)` |
| `cost_analysis_.RevisitInstruction(instr)` | `cost_model_->Revisit(instr)` |
| `fusion_analysis_cache_.Get(...)` in `CanFuseTriton` / reduce-fusion-kind check | **Remove.** This block moves to GPU's `BackendCanFuse`. |
| `CanEmitInputFusedScatter`, `FusionFitsInBudget`, `CanFuseTriton`, `IsGenericTritonFusion` | **Remove.** All move to GPU's `BackendCanFuse`. |

The existing `CalculateProducerPriority` does incremental arithmetic on cached `current_priority`. Rewrite to use `cost_model_->EstimateRunTimes` but preserve the delta-style math field-by-field. Explicit code:

```cpp
absl::Duration PriorityFusionQueue::CalculateProducerPriority(
    HloInstruction* producer) {
  // Short-circuits (stay as constants in the base pass).
  if (producer->opcode() == HloOpcode::kBitcast) return absl::InfiniteDuration();
  if (producer->opcode() == HloOpcode::kConstant) return -absl::InfiniteDuration();

  // Incremental path: only recompute over consumers that newly appeared.
  auto new_consumers_it = operands_to_new_consumers_.find(producer);
  const bool has_cached =
      reverse_map_.contains(producer) && new_consumers_it != operands_to_new_consumers_.end();

  if (has_cached) {
    absl::Span<HloInstruction* const> new_consumers = new_consumers_it->second;
    FusionCostModel::RunTimes new_rt =
        cost_model_->EstimateRunTimes(producer, new_consumers);

    // Pull the cached previous priority and the sum of removed consumers'
    // runtimes (field-by-field — do NOT collapse to a single delta).
    // The priority queue key is `std::pair<absl::Duration, int64_t>`;
    // `.first.first` is the cached priority.
    absl::Duration cached_priority = reverse_map_.at(producer)->first.first;
    auto removed_it = operands_to_removed_consumers_runtimes_.find(producer);
    FusionCostModel::RunTimes removed_rt =
        (removed_it != operands_to_removed_consumers_runtimes_.end())
            ? removed_it->second
            : FusionCostModel::RunTimes{};

    // Subtract removed, add new. unfused and fused subtract independently.
    absl::Duration new_delta =
        (new_rt.unfused - new_rt.fused) - (removed_rt.unfused - removed_rt.fused);
    return cached_priority + new_delta;
  }

  // First time we see this producer: full compute over all users.
  FusionCostModel::RunTimes rt =
      cost_model_->EstimateRunTimes(producer, producer->users());
  return rt.unfused - rt.fused;
}
```

Key property: the `removed` map's value type is `FusionCostModel::RunTimes` (two `absl::Duration` fields), not a single `absl::Duration`. Subtracting the pair field-by-field and recombining into a delta at the end is what keeps compile-time linear per update.

- [ ] **Step 6: Rewrite `PriorityFusionQueue::CanFuse`**

The current body has ~100 lines combining generic checks and GPU-specific checks. Keep the generic half; move the GPU half to `BackendCanFuse`:

```cpp
FusionDecision PriorityFusionQueue::CanFuse(HloInstruction* producer,
                                             HloInstruction* consumer) {
  // Generic: root guard.
  if (producer == producer->parent()->root_instruction()) {
    return FusionDecision::Forbid(
        "not fusing into the output of the root instruction");
  }
  // Generic: IsFusible via pass virtual.
  if (!is_fusible_(*producer)) {
    return FusionDecision::Forbid("the producer is not fusible");
  }
  if (!is_fusible_(*consumer)) {
    return FusionDecision::Forbid("the consumer is not fusible");
  }
  // Backend-specific: Triton / bitcast / scatter / reduce / budget on GPU;
  // bitcast / concat / kflops-work-items on CPU. Bitcast is here (not generic)
  // because GPU must run Triton preference first, which can legitimately fuse
  // into bitcast consumers in some code paths.
  if (auto backend = backend_can_fuse_(producer, consumer); !backend) {
    return backend;
  }
  // Generic: IR-size budget via cost model.
  if (cost_model_->WouldExplodeIrSize(producer, consumer)) {
    return FusionDecision::Forbid(
        "the fusion would result in an overly large code duplication");
  }
  // Generic tail: aliasing / in-place safety. MUST stay — this is the
  // ShouldFuseInPlaceOp check at `priority_fusion.cc:873` in the original.
  return InstructionFusion::ShouldFuseInPlaceOp(
      producer, consumer, alias_info_, /*fusion_node_evaluations=*/std::nullopt);
}
```

`IsFusibleBitcast` was file-local in the old GPU `priority_fusion.cc` (definition was `static bool IsFusibleBitcast(...)`). v4→v5 **promotes its declaration to the shared `priority_fusion.h`** (keeping the definition in `.cc`) so both GPU and CPU subclasses — which live in different translation units — can call it. The **call** moves into each backend's `BackendCanFuse`: GPU puts it after Triton preference (preserving current semantics where Triton gets first look at bitcast consumers); CPU puts it first among its backend checks. Both subclass `.cc` files need `#include "xla/hlo/transforms/priority_fusion.h"` to see the declaration.

**Implementer note:** when reconstituting the shared `priority_fusion.cc`, **strip the `static` keyword** from the `IsFusibleBitcast` definition. With the header declaring the function in `namespace xla`, a `static` definition would be an ODR violation (internal linkage in the `.cc` vs external linkage in the header). Leave the body unchanged.

Keep `CanFuseCached` with `can_fuse_cache_` unchanged — it caches this function's result and is generic.

- [ ] **Step 7: Implement `PriorityFusion::IsFusible` default (matches today's GPU allowlist)**

In `priority_fusion.cc`:

```cpp
bool PriorityFusion::IsFusible(const HloInstruction& instr) {
  // Matches the free-function `IsFusible` in the old file
  // (priority_fusion.cc:107–132). Move that body here verbatim.
  if (!instr.IsFusible()) return false;
  if (instr.opcode() == HloOpcode::kFusion) return true;
  switch (instr.opcode()) {
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kPad:
    case HloOpcode::kGather:
    case HloOpcode::kScatter:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kReduce:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCopy:
    case HloOpcode::kIota:
    case HloOpcode::kConstant:
      return true;
    default:
      return instr.IsElementwise();
  }
}
```

Delete the free-function `IsFusible` that's currently in the file.

- [ ] **Step 8: Wire `PriorityFusion::Run` to construct the queue**

In `PriorityFusion::Run`, for each computation:

```cpp
// Prepare the cost model FIRST. GPU uses this to run its initial
// HloCostAnalysis Accept(). CPU uses it similarly. Before v3 this
// was done implicitly inside PriorityFusionQueue's constructor;
// after the move it must be called explicitly.
TF_RETURN_IF_ERROR(cost_model_->Prepare(computation));

PriorityFusionQueue queue(
    computation,
    cost_model_.get(),
    /*backend_can_fuse=*/
    [this](HloInstruction* p, HloInstruction* c) {
      return BackendCanFuse(p, c);
    },
    alias_info_,
    thread_pool_,
    fusion_process_dump_.get(),
    /*is_fusible=*/
    [this](const HloInstruction& i) { return IsFusible(i); });
```

After each fusion, call `cost_model_->OnInstructionFused(producer, consumer, fusion)` from **inside `PriorityFusion::Fuse()`** — at the tail, right before the `return fusion;` line (old location around `priority_fusion.cc:~892`). This keeps the hook invariant consistent regardless of which caller initiated the fusion.

- [ ] **Step 9: Update the existing BUILD target (in place)**

In `xla/backends/gpu/transforms/BUILD`, the existing `priority_fusion` `cc_library` target stays. **Add** a new dep `//xla/hlo/transforms:fusion_cost_model`. GPU-specific deps (`//xla/service/gpu/model:gpu_hlo_cost_analysis`, etc.) can be removed once Task 0.3 lands — for now leave them stale to keep this task self-contained.

Run: `bazel build //xla/backends/gpu/transforms:priority_fusion`
Expected: success after this task's code changes.

- [ ] **Step 10: Run the existing test suite**

```bash
bazel test //xla/backends/gpu/transforms:priority_fusion_test
```

This requires Task 0.3's `GpuFusionCostModel` stub to already exist, because the queue now needs *some* `FusionCostModel*` to instantiate. **Order of operations:** land Task 0.3's minimal stub (header + empty overrides returning sentinel values) BEFORE finishing Task 0.2, so tests still pass. Alternative: commit Task 0.2 + Task 0.3 together as a single commit (acceptable fallback if a stub is awkward).

- [ ] **Step 11: Commit**

```bash
git add -A xla/backends/gpu/transforms xla/hlo/transforms
git commit -m "Refactor PriorityFusionQueue to use FusionCostModel interface (in place)"
```

---

### Task 0.3: Create `GpuFusionCostModel` and `GpuPriorityFusion`

**Files:**
- Create: `xla/backends/gpu/transforms/gpu_fusion_cost_model.h`
- Create: `xla/backends/gpu/transforms/gpu_fusion_cost_model.cc`
- Create: `xla/backends/gpu/transforms/gpu_priority_fusion.h`
- Create: `xla/backends/gpu/transforms/gpu_priority_fusion.cc`
- Modify: `xla/backends/gpu/transforms/BUILD`

- [ ] **Step 1: `gpu_fusion_cost_model.h`**

```cpp
/* Copyright 2026 The OpenXLA Authors. [standard header] */

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_GPU_FUSION_COST_MODEL_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_GPU_FUSION_COST_MODEL_H_

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/fusion_cost_model.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/fusion_deduplication_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// FusionCostModel for GPU. Owns every GPU-specific cost-analysis cache
// that used to live inside PriorityFusionQueue: GpuHloCostAnalysis,
// GpuPerformanceModel (+ its cache), HloFusionAnalysisCache,
// FusionDeduplicationCache, the tiled run-time data cache, and the
// MLIRContext that the tiled model uses.
class GpuFusionCostModel : public FusionCostModel {
 public:
  GpuFusionCostModel(const se::DeviceDescription& device_info,
                     GpuHloCostAnalysis::Options cost_analysis_options,
                     mlir::MLIRContext* mlir_context);

  RunTimes EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<HloInstruction* const> consumers) override;
  bool WouldExplodeIrSize(const HloInstruction* producer,
                          const HloInstruction* consumer) override;
  void OnInstructionFused(HloInstruction* producer, HloInstruction* consumer,
                          HloInstruction* fusion) override;
  void Invalidate(const HloInstruction* instruction) override;
  absl::Status Revisit(const HloInstruction* instruction) override;

  // Accessors used by GpuPriorityFusion::BackendCanFuse.
  const se::DeviceDescription& device_info() const { return device_info_; }
  HloFusionAnalysisCache& fusion_analysis_cache() { return fusion_analysis_cache_; }
  GpuHloCostAnalysis& cost_analysis() { return cost_analysis_; }
  mlir::MLIRContext* mlir_context() { return mlir_context_; }
  FusionDeduplicationCache& fusion_deduplication_cache() {
    return fusion_deduplication_cache_;
  }

  // Prepare override (interface method). Accepts() the computation
  // into cost_analysis_ and populates internal caches. Called by the
  // base pass's PriorityFusion::Run before the queue is constructed.
  absl::Status Prepare(HloComputation* computation) override;

 private:
  // Private helper for Triton tiling cost — fold of old
  // PriorityFusionQueue::GetTiledRunTimeDataCached. Lives here so the
  // shared queue never sees TiledRunTimeDataOrError or
  // BlockLevelParameters.
  absl::StatusOr<TiledRunTimeDataOrError> GetTiledRunTimeDataCached(
      const HloInstruction* producer, const HloInstruction* consumer);

  se::DeviceDescription device_info_;
  GpuHloCostAnalysis cost_analysis_;
  GpuPerformanceModelCache gpu_performance_model_cache_;
  HloFusionAnalysisCache fusion_analysis_cache_;
  GpuPerformanceModel gpu_performance_model_;
  GpuIndexingPerformanceModel gpu_indexing_performance_model_;
  FusionDeduplicationCache fusion_deduplication_cache_;
  mlir::MLIRContext* mlir_context_;
  // Tiled Triton run-time data cache. Consulted internally by
  // EstimateRunTimes; never exposed through the interface.
  absl::flat_hash_map<std::pair<const HloInstruction*, const HloInstruction*>,
                      TiledRunTimeDataOrError>
      tiled_run_time_data_cache_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_GPU_FUSION_COST_MODEL_H_
```

- [ ] **Step 2: `gpu_fusion_cost_model.cc` — move the logic from the old queue**

Implement each interface method by lifting the corresponding code out of the old `PriorityFusionQueue`:

- `Prepare(HloComputation*)` — initial cost-analysis and cache setup from the old `PriorityFusionQueue` constructor body: `TF_RETURN_IF_ERROR(computation->Accept(&cost_analysis_)); fusion_analysis_cache_.Invalidate(); ...`. Returns `absl::OkStatus()` on success.
- `EstimateRunTimes` — the body of the old `CalculateProducerPriority` that called `gpu_performance_model_.EstimateRunTimes(producer, &cost_analysis_, cost_analysis_options, consumers)`. Copy lines around `priority_fusion.cc:584` and `613`. **Also includes the tiled-data Triton path** that the old queue's `CalculateProducerPriority` reached via `GetTiledRunTimeDataCached` (old lines ~664–710, ~748). That entire sub-path is now private to this method. Return the `{unfused, fused}` pair directly; the base pass does the subtraction.
- `WouldExplodeIrSize` — forward to `cost_analysis_.ProducerConsumerMergedTooLarge(*producer, *consumer)`.
- `OnInstructionFused` — update `gpu_performance_model_cache_` per the old queue logic around `priority_fusion.cc:760`. Also erase any `tiled_run_time_data_cache_` entries keyed on the fused-away instructions.
- `Invalidate` — the old queue's `priority_fusion.cc:360–368` body: `can_fuse_cache_.erase(instr)` was queue-local (stays in queue); `gpu_performance_model_cache_.Invalidate(*instr)` moves here. Also erase the instruction's `tiled_run_time_data_cache_` entries.
- `Revisit` — forward to `cost_analysis_.RevisitInstruction(instr)`.
- `GetTiledRunTimeDataCached` — lift verbatim from old `priority_fusion.cc:664–710` as a private method here.

- [ ] **Step 3: `gpu_priority_fusion.h`**

```cpp
/* Copyright 2026 The OpenXLA Authors. [standard header] */

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_GPU_PRIORITY_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_GPU_PRIORITY_FUSION_H_

#include <memory>
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/transforms/gpu_fusion_cost_model.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"  // base still here in Phase 0; Phase A will rewrite this include
#include "xla/service/gpu/fusion_info_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {

// Thin subclass of the shared PriorityFusion for GPU. Constructor
// signature matches the old `gpu::PriorityFusion` so existing call
// sites only change the type name.
class GpuPriorityFusion : public PriorityFusion {
 public:
  GpuPriorityFusion(tsl::thread::ThreadPool* thread_pool,
                    const se::DeviceDescription& device,
                    const AliasInfo* alias_info,
                    GpuHloCostAnalysis::Options cost_analysis_options,
                    mlir::MLIRContext* mlir_context);

  // NOTE: GpuPriorityFusion intentionally does NOT override IsFusible.
  // The base-class default is the GPU allowlist by convention — any
  // future divergence (e.g., if GPU starts supporting new opcodes
  // differently from CPU) requires an explicit override here, not a
  // silent edit to the base.

 protected:
  FusionDecision BackendCanFuse(HloInstruction* producer,
                                HloInstruction* consumer) override;
  HloInstruction::FusionKind ChooseKind(const HloInstruction* producer,
                                         const HloInstruction* consumer) override;

 private:
  // Non-owning pointer to the concrete cost model installed in the base
  // class. Used by BackendCanFuse to reach Triton caches, device info,
  // etc.
  GpuFusionCostModel* gpu_cost_model_;
  FusionInfoCache fusion_info_cache_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_GPU_PRIORITY_FUSION_H_
```

- [ ] **Step 4: `gpu_priority_fusion.cc`**

Constructor:

```cpp
GpuPriorityFusion::GpuPriorityFusion(
    tsl::thread::ThreadPool* thread_pool,
    const se::DeviceDescription& device,
    const AliasInfo* alias_info,
    GpuHloCostAnalysis::Options cost_analysis_options,
    mlir::MLIRContext* mlir_context)
    : PriorityFusion(thread_pool, alias_info,
                     std::make_unique<GpuFusionCostModel>(
                         device, std::move(cost_analysis_options),
                         mlir_context)) {
  gpu_cost_model_ = static_cast<GpuFusionCostModel*>(cost_model());
}
```

`BackendCanFuse` — paste the old `CanFuse` blocks in order (lines ~790–862 of old `priority_fusion.cc`):

```cpp
FusionDecision GpuPriorityFusion::BackendCanFuse(HloInstruction* producer,
                                                  HloInstruction* consumer) {
  // 1. Triton preference — kept FIRST so Triton can intercept
  //    bitcast-consumer cases before the generic bitcast forbid.
  FusionDecision can_fuse_triton = CanFuseTriton(producer, consumer);
  if (IsGenericTritonFusion(*producer) || IsGenericTritonFusion(*consumer) ||
      can_fuse_triton) {
    return can_fuse_triton;
  }
  // 2. Bitcast-only consumer forbid. Preserves the original
  //    CanFuse ordering at priority_fusion.cc:803 (immediately after
  //    the Triton block).
  if (IsFusibleBitcast(*consumer)) {
    return FusionDecision::Forbid(
        "not fusing into a single bitcast as consumer");
  }
  // 3. CanEmitInputFusedScatter — legality check only.
  if (auto can_fuse = CanEmitInputFusedScatter(*producer, *consumer);
      !can_fuse) {
    return can_fuse;
  }
  // 4. Significant reduce into reduce.
  auto contains_significant_reduce = [&](const HloInstruction* instr) {
    auto fusion = HloFusionAdaptor::ForInstruction(instr);
    return HloAnyOf(*fusion, [](auto node) {
      if (!(node.opcode() == HloOpcode::kReduce && node.shape().IsArray())) {
        return false;
      }
      int64_t reduction_size =
          ShapeUtil::ElementsIn(node.instruction().operand(0)->shape()) /
          ShapeUtil::ElementsIn(node.shape());
      return reduction_size >= 16;
    });
  };
  if (contains_significant_reduce(producer) &&
      contains_significant_reduce(consumer)) {
    return FusionDecision::Forbid(
        "both the producer and the consumer contain a reduce");
  }
  // 5. Reduce → loop switch avoidance via fusion_analysis_cache_.
  const auto& analysis = gpu_cost_model_->fusion_analysis_cache().Get(*producer);
  if (analysis.emitter_fusion_kind() ==
      HloFusionAnalysis::EmitterFusionKind::kReduction) {
    const auto& analysis_fused =
        gpu_cost_model_->fusion_analysis_cache().Get(*producer, *consumer);
    if (analysis_fused.emitter_fusion_kind() ==
        HloFusionAnalysis::EmitterFusionKind::kLoop) {
      return FusionDecision::Forbid(
          "fusion into output of a reduce fusion would create a loop fusion");
    }
  }
  // 6. FusionFitsInBudget.
  if (auto fits_budget = FusionFitsInBudget(
          *consumer, *producer, gpu_cost_model_->device_info(),
          /*is_consumer_producer_fusion=*/true, &fusion_info_cache_);
      !fits_budget) {
    return fits_budget;
  }
  return FusionDecision::Allow();
}
```

`ChooseKind` — paste the old `ChooseKind` body verbatim.

`CanFuseTriton` — this was a private method on `PriorityFusionQueue`. Move it into this file as a private static helper; it takes `gpu_cost_model_->mlir_context()` etc. as arguments.

- [ ] **Step 5: Update `xla/backends/gpu/transforms/BUILD`**

Add targets for the two new files:

```python
cc_library(
    name = "gpu_fusion_cost_model",
    srcs = ["gpu_fusion_cost_model.cc"],
    hdrs = ["gpu_fusion_cost_model.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms:fusion_cost_model",
        "//xla/service/gpu/model:fusion_analysis_cache",
        "//xla/service/gpu/model:fusion_deduplication_cache",
        "//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "//xla/service/gpu/model:gpu_indexing_performance_model",
        "//xla/service/gpu/model:gpu_performance_model",
        "//xla/service/gpu/model:gpu_performance_model_base",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "gpu_priority_fusion",
    srcs = ["gpu_priority_fusion.cc"],
    hdrs = ["gpu_priority_fusion.h"],
    deps = [
        ":gpu_fusion_cost_model",
        "//xla:shape_util",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms:priority_fusion",
        "//xla/service/gpu:fusion_info_cache",
        "//xla/service/gpu:triton_support",
        "//xla/service/gpu/model:fusion_analysis_cache",
        "//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:threadpool",
        "@llvm-project//mlir:IR",
    ],
)
```

Verify with `bazel query 'deps(//xla/backends/gpu/transforms:gpu_priority_fusion)'`.

---

### Task 0.4: Update GPU call sites (type rename)

**Files (one edit each, plus each file's BUILD):**
- `xla/service/gpu/fusion_pipeline.cc:76`
- `xla/backends/gpu/transforms/triton_fusion_numerics_verifier.cc:~136`
- `xla/backends/gpu/autotuner/triton.cc:~276`
- `xla/backends/gpu/autotuner/fission_backend.cc:~127`

- [ ] **Step 1: For each file above:**
  - Change `#include "xla/backends/gpu/transforms/priority_fusion.h"` → `#include "xla/backends/gpu/transforms/gpu_priority_fusion.h"`.
  - Change `PriorityFusion(...)` or `gpu::PriorityFusion(...)` → `GpuPriorityFusion(...)`. Constructor signature is identical; only the type name changes.
  - In the matching `BUILD` file, change `//xla/backends/gpu/transforms:priority_fusion` → `//xla/backends/gpu/transforms:gpu_priority_fusion`.

- [ ] **Step 2: Build the whole GPU fusion subtree**

```bash
bazel build //xla/service/gpu:fusion_pipeline
bazel build //xla/backends/gpu/transforms:gpu_priority_fusion
bazel build //xla/backends/gpu/autotuner:triton
bazel build //xla/backends/gpu/autotuner:fission_backend
bazel build //xla/backends/gpu/transforms:triton_fusion_numerics_verifier
```

Expected: all succeed. Fix any build errors before proceeding.

- [ ] **Step 3: Commit Task 0.4**

```bash
git add -A xla/service/gpu/fusion_pipeline.cc xla/backends/gpu/autotuner xla/backends/gpu/transforms
git commit -m "Rename GPU PriorityFusion callers to GpuPriorityFusion (in place)

Updates 4 call sites: fusion_pipeline.cc, triton_fusion_numerics_verifier,
autotuner/triton, autotuner/fission_backend. Files remain at
xla/backends/gpu/transforms/; this is the rename commit. Phase A later
moves the base pass to xla/hlo/transforms/."
```

---

### Task 0.5: Add FakeCostModel-driven base-pass test (in place)

**Files:**
- Modify: `xla/backends/gpu/transforms/priority_fusion_test.cc` (add new test section; existing tests unchanged)
- Modify: `xla/backends/gpu/transforms/BUILD` (add `//xla/hlo/transforms:fusion_cost_model` dep)

The test file stays at its current GPU location. Phase A relocates it later.

- [ ] **Step 1: Update type references in existing tests**

The existing tests used `PriorityFusion` directly; now the queue's constructor takes a `FusionCostModel*`. Either:
- (a) keep constructing `gpu::GpuPriorityFusion` (Task 0.3 introduced it) — matches the new public construction pattern; **recommended**.
- (b) construct a `PriorityFusion` directly with a hand-built `GpuFusionCostModel` — more plumbing, no benefit.

Update every `PriorityFusion pass(...)` to `GpuPriorityFusion pass(...)`. Signature is identical — existing tests need only a type rename.

- [ ] **Step 2: Add a `FakeCostModel`-driven base-pass test at the end of the test file**

```cpp
namespace base_pass_test {

class FakeCostModel : public ::xla::FusionCostModel {
 public:
  RunTimes EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<HloInstruction* const> consumers) override {
    // Deterministic: fusion always beneficial (unfused > fused),
    // ordered by unique_id (larger id → smaller benefit → lower priority).
    return {/*unfused=*/absl::Microseconds(1000 + producer->unique_id()),
            /*fused=*/absl::Microseconds(500)};
  }
  bool WouldExplodeIrSize(const HloInstruction* /*p*/,
                          const HloInstruction* /*c*/) override {
    return false;
  }
};

TEST_F(PriorityFusionTest, BaseQueueFusesAllFusibleWithFakeCostModel) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      p0 = f32[1024] parameter(0)
      a = f32[1024] add(p0, p0)
      b = f32[1024] multiply(a, a)
      ROOT r = f32[1024] subtract(b, p0)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(std::string(kHlo)));
  ::xla::AliasInfo alias_info;
  // NOTE: uses the bare base PriorityFusion (not GpuPriorityFusion) with
  // the fake cost model — proves the base pass has no GPU dependency.
  ::xla::gpu::PriorityFusion pass(
      /*thread_pool=*/nullptr, &alias_info, std::make_unique<FakeCostModel>());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  // Verify some fusion happened.
}

}  // namespace base_pass_test
```

Note: `::xla::gpu::PriorityFusion` because the base pass is still in the `xla::gpu` namespace during Phase 0. Phase A changes this to `::xla::PriorityFusion`.

- [ ] **Step 3: Update the existing test target's deps**

In `xla/backends/gpu/transforms/BUILD`, the `priority_fusion_test` `cc_test` target gains `//xla/hlo/transforms:fusion_cost_model`. No new target needed.

- [ ] **Step 4: Run the tests**

```bash
bazel test //xla/backends/gpu/transforms:priority_fusion_test
```

Expected: all existing GPU tests pass + the new `FakeCostModel` test passes. Any failure in the pre-existing tests means Phase 0's prior tasks drifted behavior.

- [ ] **Step 5: Commit**

```bash
git add -A xla/backends/gpu/transforms/priority_fusion_test.cc xla/backends/gpu/transforms/BUILD
git commit -m "Add FakeCostModel-driven base-pass test (in place)

Proves PriorityFusion + PriorityFusionQueue work with a trivial
non-GPU cost-model implementation. The test lives next to GPU tests
until Phase A relocates base pass files to xla/hlo/transforms/."
```

---

## Phase A — File relocation (mechanical)

At this point every file in `xla/backends/gpu/transforms/` that will move is fully decoupled from GPU types. Phase A is a pure mechanical relocation: move the files, change namespace/guards, update BUILD paths, update includes in the 4 GPU callers. Zero behavior change.

**Phase A commit strategy.** Each task commits standalone. Because the files are already decoupled, there is no broken-intermediate-state problem. Worst case the build breaks transiently during a `git mv`; the next task fixes it.

### Task A.1: Move base pass + queue files to shared location

**Files:**
- Move: `xla/backends/gpu/transforms/priority_fusion.{h,cc}` → `xla/hlo/transforms/priority_fusion.{h,cc}`
- Modify: `xla/hlo/transforms/BUILD` (add new `priority_fusion` target)
- Modify: `xla/backends/gpu/transforms/BUILD` (remove the `priority_fusion` `cc_library` target)

- [ ] **Step 1: `git mv` the files**

```bash
git mv xla/backends/gpu/transforms/priority_fusion.h \
       xla/hlo/transforms/priority_fusion.h
git mv xla/backends/gpu/transforms/priority_fusion.cc \
       xla/hlo/transforms/priority_fusion.cc
```

- [ ] **Step 2: Change namespace and include guard**

In the header:
- `XLA_BACKENDS_GPU_TRANSFORMS_PRIORITY_FUSION_H_` → `XLA_HLO_TRANSFORMS_PRIORITY_FUSION_H_` (include guard + `#endif` comment).
- `namespace xla { namespace gpu {` → `namespace xla {` (drop nested gpu).
- `}  // namespace gpu  }  // namespace xla` → `}  // namespace xla`.

In the `.cc`: mirror the namespace change. Update the own-header `#include` to the new path.

- [ ] **Step 3: Promote `IsFusibleBitcast`**

`IsFusibleBitcast` was `static` file-local in the old `.cc` and already declared in the (now-moved) header per Task 0.2 Step 2. Strip the `static` keyword from the definition in the `.cc` (header declares external linkage; a `static` definition would be an ODR violation).

- [ ] **Step 4: Move BUILD target**

Remove the `priority_fusion` `cc_library` from `xla/backends/gpu/transforms/BUILD`. Add it at `xla/hlo/transforms/BUILD`:

```python
cc_library(
    name = "priority_fusion",
    srcs = ["priority_fusion.cc"],
    hdrs = ["priority_fusion.h"],
    deps = [
        ":fusion_cost_model",
        "//xla:shape_util",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:fusion_process_dump_proto_cc",
        "//xla/service:instruction_fusion",
        "//xla/tsl/platform:threadpool",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
```

Verify:
```bash
bazel query 'deps(//xla/hlo/transforms:priority_fusion)' | head -20
bazel build //xla/hlo/transforms:priority_fusion
```

- [ ] **Step 5: Update the 4 GPU callers' includes**

Each of `fusion_pipeline.cc`, `triton_fusion_numerics_verifier.cc`, `autotuner/triton.cc`, `autotuner/fission_backend.cc` — plus the `gpu_priority_fusion.h` header — change:

```cpp
// Before:
#include "xla/backends/gpu/transforms/priority_fusion.h"
// After:
#include "xla/hlo/transforms/priority_fusion.h"
```

Type names were already changed to `GpuPriorityFusion` in Task 0.4. Only the include path changes here.

Each caller's BUILD dep changes: `//xla/backends/gpu/transforms:priority_fusion` → `//xla/hlo/transforms:priority_fusion`.

- [ ] **Step 6: Build everything**

```bash
bazel build //xla/service/gpu:fusion_pipeline
bazel build //xla/backends/gpu/transforms:gpu_priority_fusion
bazel build //xla/backends/gpu/autotuner:triton
bazel build //xla/backends/gpu/autotuner:fission_backend
```

Expected: all succeed. If anything fails, revert `git mv` and debug.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Move PriorityFusion + PriorityFusionQueue to xla/hlo/transforms/

Pure file relocation. Base pass is already fully decoupled from GPU
types (Phase 0). Only namespace (xla::gpu -> xla), include guards,
include paths, and BUILD target locations change. Strip `static` from
IsFusibleBitcast definition (now declared in header). Zero logic
change."
```

### Task A.2: Move `priority_fusion_test.cc` to shared location

**Files:**
- Move: `xla/backends/gpu/transforms/priority_fusion_test.cc` → `xla/hlo/transforms/priority_fusion_test.cc`
- Modify: `xla/hlo/transforms/BUILD` (add test target)
- Modify: `xla/backends/gpu/transforms/BUILD` (remove old test target)

- [ ] **Step 1: `git mv` the file**

```bash
git mv xla/backends/gpu/transforms/priority_fusion_test.cc \
       xla/hlo/transforms/priority_fusion_test.cc
```

- [ ] **Step 2: Update includes and namespace references**

- `#include "xla/backends/gpu/transforms/priority_fusion.h"` → `#include "xla/hlo/transforms/priority_fusion.h"` (base header moved in A.1).
- In the `FakeCostModel` test: `::xla::gpu::PriorityFusion` → `::xla::PriorityFusion`.
- GPU-subclass usages (`GpuPriorityFusion`) keep their `::xla::gpu::` qualification — that subclass did not move.
- Pre-existing test `namespace xla::gpu` can stay (test is still GPU-specific), or flatten to `namespace xla::gpu::priority_fusion_test` if preferred.

- [ ] **Step 3: Add the test target in the new BUILD**

```python
xla_cc_test(
    name = "priority_fusion_test",
    srcs = ["priority_fusion_test.cc"],
    deps = [
        ":fusion_cost_model",
        ":priority_fusion",
        "//xla/backends/gpu/transforms:gpu_priority_fusion",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        # Copy remaining deps from the old priority_fusion_test target.
    ],
)
```

Add `load("//xla:xla.bzl", "xla_cc_test")` at the top of the BUILD file.

Remove the old `priority_fusion_test` target from `xla/backends/gpu/transforms/BUILD`.

- [ ] **Step 4: Run the tests**

```bash
bazel test //xla/hlo/transforms:priority_fusion_test
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A xla/hlo/transforms xla/backends/gpu/transforms/BUILD
git commit -m "Move priority_fusion_test to xla/hlo/transforms/"
```

---

---

## Phase B — CPU cost model (TDD)

### Task B1: Skeleton of `CpuFusionCostModel`

**Files:**
- Create: `xla/service/cpu/cpu_fusion_cost_model.h`
- Create: `xla/service/cpu/cpu_fusion_cost_model.cc`
- Modify: `xla/service/cpu/BUILD`

- [ ] **Step 1: Header**

```cpp
/* Copyright 2026 The OpenXLA Authors. [standard header] */

#ifndef XLA_SERVICE_CPU_CPU_FUSION_COST_MODEL_H_
#define XLA_SERVICE_CPU_CPU_FUSION_COST_MODEL_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/fusion_cost_model.h"
#include "xla/service/fusion_node_indexing_evaluation.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"

namespace xla::cpu {

struct ComplexOpPenalties {
  int64_t concatenate_kflops_per_element = 4;
  int64_t pad_kflops_per_element = 2;
  int64_t dynamic_slice_kflops_per_element = 2;
  int64_t dynamic_update_slice_kflops_per_element = 3;
};

struct CpuDeviceInfo {
  int64_t num_threads = 1;
  double peak_kflops_per_second = 1e9;    // ~1 GFLOPS per thread, summed
  double peak_bytes_per_second = 1e10;    // ~10 GB/s aggregate
};

class CpuFusionCostModel : public FusionCostModel {
 public:
  // `hlo_cost_analysis` is NOT Accept()-ed here. Prepare() does it.
  // `precise_invalidation` controls OnInstructionFused: true (default,
  // recommended) erases only the consumer's fusion_node_evaluations_
  // entry; false clears the whole map (O(N²), diagnosis-only).
  // Default = true for `precise_invalidation` so existing test
  // construction sites (Phase B tests pass only 3 args) still compile.
  CpuFusionCostModel(HloCostAnalysis* hlo_cost_analysis,
                     CpuDeviceInfo device_info,
                     ComplexOpPenalties penalties,
                     bool precise_invalidation = true);

  // FusionCostModel overrides.
  absl::Status Prepare(HloComputation* computation) override;
  RunTimes EstimateRunTimes(
      const HloInstruction* producer,
      absl::Span<HloInstruction* const> consumers) override;
  bool WouldExplodeIrSize(const HloInstruction* producer,
                          const HloInstruction* consumer) override;
  void OnInstructionFused(HloInstruction* producer,
                          HloInstruction* consumer,
                          HloInstruction* fusion) override;

  // CPU-only helpers used by CpuPriorityFusion::BackendCanFuse.
  int64_t Kflops(const HloInstruction& instr) const;
  int64_t IndependentWorkItems(const HloInstruction& instr) const;
  int64_t FusedKflops(const HloInstruction& producer,
                      const HloInstruction& consumer) const;
  int64_t FusedWorkItems(const HloInstruction& producer,
                         const HloInstruction& consumer) const;

 private:
  int64_t ComplexOpKflops(const HloInstruction& instr) const;
  int64_t WorkItemsFromShape(const Shape& shape) const;

  // Non-const pointer so Prepare() can call Accept() on it.
  HloCostAnalysis* hlo_cost_analysis_;  // not owned
  CpuDeviceInfo device_info_;
  ComplexOpPenalties penalties_;
  bool precise_invalidation_;

  // Lazy, keyed on the `consumer` passed to WouldExplodeIrSize. Mirrors
  // CpuInstructionFusion's fusion_node_evaluations_.
  mutable absl::flat_hash_map<const HloInstruction*,
                               FusionNodeIndexingEvaluation>
      fusion_node_evaluations_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_FUSION_COST_MODEL_H_
```

- [ ] **Step 2: `.cc` skeleton — every method returns `LOG(FATAL)` except the ctor and `Prepare`**

`Prepare` has a concrete implementation from day one because otherwise the cost analysis stays empty and every Kflops returns 0:

```cpp
absl::Status CpuFusionCostModel::Prepare(HloComputation* computation) {
  return computation->Accept(hlo_cost_analysis_);
}
```

Every other method is `LOG(FATAL) << "not implemented";` until filled in by subsequent B-tasks.

- [ ] **Step 3: BUILD target**

In `xla/service/cpu/BUILD`:

```python
cc_library(
    name = "cpu_fusion_cost_model",
    srcs = ["cpu_fusion_cost_model.cc"],
    hdrs = ["cpu_fusion_cost_model.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms:fusion_cost_model",
        "//xla/service:fusion_node_indexing_evaluation",
        "//xla/service:hlo_cost_analysis",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)
```

- [ ] **Step 4: Build + commit**

```bash
bazel build //xla/service/cpu:cpu_fusion_cost_model
git add xla/service/cpu/cpu_fusion_cost_model.{h,cc} xla/service/cpu/BUILD
git commit -m "Add CpuFusionCostModel skeleton"
```

---

### Task B2: Implement `Kflops` for ordinary (non-complex) ops

- [ ] **Step 1: Write failing test**

Create `xla/service/cpu/cpu_fusion_cost_model_test.cc`:

```cpp
#include "xla/service/cpu/cpu_fusion_cost_model.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

class CpuFusionCostModelTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<HloCostAnalysis> MakeAnalysis(HloModule* module) {
    HloCostAnalysis::Options options;
    options.shape_size = [](const Shape& s) {
      return CpuExecutable::ShapeSizeBytes(s);
    };
    auto analysis = std::make_unique<HloCostAnalysis>(options);
    CHECK_OK(module->entry_computation()->Accept(analysis.get()));
    return analysis;
  }
};

TEST_F(CpuFusionCostModelTest, KflopsForElementwiseAddUsesHloCostAnalysis) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      p0 = f32[1024] parameter(0)
      ROOT a = f32[1024] add(p0, p0)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(std::string(kHlo)));
  auto analysis = MakeAnalysis(module.get());
  CpuFusionCostModel model(analysis.get(), {}, {});
  const HloInstruction* add = module->entry_computation()->root_instruction();
  // f32[1024] add: 1024 flops / 1000 = 1 kflop.
  EXPECT_EQ(model.Kflops(*add), 1);
}

}  // namespace
}  // namespace xla::cpu
```

Add BUILD test target:
```python
xla_cc_test(
    name = "cpu_fusion_cost_model_test",
    srcs = ["cpu_fusion_cost_model_test.cc"],
    deps = [
        ":cpu_fusion_cost_model",
        ":cpu_executable",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:hlo_cost_analysis",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
```

- [ ] **Step 2: Run — FAIL**

```bash
bazel test //xla/service/cpu:cpu_fusion_cost_model_test \
  --test_filter="CpuFusionCostModelTest.KflopsForElementwiseAddUsesHloCostAnalysis"
```

Expected: FAIL (LOG(FATAL)).

- [ ] **Step 3: Implement**

```cpp
int64_t CpuFusionCostModel::Kflops(const HloInstruction& instr) const {
  switch (instr.opcode()) {
    case HloOpcode::kConcatenate:
    case HloOpcode::kPad:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
      return ComplexOpKflops(instr);
    default:
      break;
  }
  int64_t flops = static_cast<int64_t>(hlo_cost_analysis_->flop_count(instr));
  return flops / 1000;
}

int64_t CpuFusionCostModel::ComplexOpKflops(const HloInstruction& /*instr*/)
    const {
  LOG(FATAL) << "not implemented; see Task B3";
}
```

- [ ] **Step 4: Run — PASS. Commit.**

---

### Task B3: Implement complex-op penalties (4 sub-tests)

- [ ] **Step 1: Add 4 failing tests**

```cpp
TEST_F(CpuFusionCostModelTest, KflopsForConcatenateUsesPenalty) {
  // Major-dim concat, f32[128,128] = 16384 elems * 4 / 1000 = 65 kflops.
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      p0 = f32[64,128] parameter(0)
      p1 = f32[64,128] parameter(1)
      ROOT c = f32[128,128] concatenate(p0, p1), dimensions={0}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(std::string(kHlo)));
  auto analysis = MakeAnalysis(module.get());
  CpuFusionCostModel model(analysis.get(), {}, {});
  EXPECT_EQ(model.Kflops(*module->entry_computation()->root_instruction()), 65);
}

TEST_F(CpuFusionCostModelTest, KflopsForPadUsesPenalty) {
  // 1024 elems * 2 / 1000 = 2 kflops (rounds down — accept 2 or 0).
}
TEST_F(CpuFusionCostModelTest, KflopsForDynamicSliceUsesPenalty) { /* … */ }
TEST_F(CpuFusionCostModelTest, KflopsForDynamicUpdateSliceUsesPenalty) { /* … */ }
```

- [ ] **Step 2: Run — FAIL**

```bash
bazel test //xla/service/cpu:cpu_fusion_cost_model_test \
  --test_filter="CpuFusionCostModelTest.Kflops*UsesPenalty"
```

Expected: FAIL.

- [ ] **Step 3: Implement**

```cpp
int64_t CpuFusionCostModel::ComplexOpKflops(const HloInstruction& instr) const {
  int64_t num_elements = ShapeUtil::ElementsIn(instr.shape());
  int64_t per_elem = 0;
  switch (instr.opcode()) {
    case HloOpcode::kConcatenate:
      per_elem = penalties_.concatenate_kflops_per_element;
      break;
    case HloOpcode::kPad:
      per_elem = penalties_.pad_kflops_per_element;
      break;
    case HloOpcode::kDynamicSlice:
      per_elem = penalties_.dynamic_slice_kflops_per_element;
      break;
    case HloOpcode::kDynamicUpdateSlice:
      per_elem = penalties_.dynamic_update_slice_kflops_per_element;
      break;
    default:
      LOG(FATAL) << "not a complex op: " << instr.opcode();
  }
  return num_elements * per_elem / 1000;
}
```

**Note:** `concat_minor_dim_multiplier` is intentionally not implemented — the hard block in `CpuPriorityFusion::BackendCanFuse` makes it unreachable today.

- [ ] **Step 4: Run — PASS. Commit.**

---

### Task B4: Implement `IndependentWorkItems` (physical outer-dim, layout-safe)

- [ ] **Step 1: Write failing tests**

```cpp
TEST_F(CpuFusionCostModelTest, WorkItemsForArrayWithLayoutUsesPhysicalOuterDim) {
  // f32[1024, 4]{0,1} — physical outer is dim with smallest stride = dim 1 → 4.
  // f32[1024, 4]{1,0} — physical outer = dim 0 → 1024.
}
TEST_F(CpuFusionCostModelTest, WorkItemsForArrayWithoutLayoutFallsBack) {
  // Parse without layout assignment; return dimensions(0).
}
TEST_F(CpuFusionCostModelTest, WorkItemsForScalarIsOne) {}
TEST_F(CpuFusionCostModelTest, WorkItemsForTupleUsesFirstElement) {
  // tuple(f32[32], f32[]) → 32, not min(32, 1) == 1.
}
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

```cpp
int64_t CpuFusionCostModel::IndependentWorkItems(
    const HloInstruction& instr) const {
  return WorkItemsFromShape(instr.shape());
}

int64_t CpuFusionCostModel::WorkItemsFromShape(const Shape& shape) const {
  if (shape.IsTuple()) {
    if (shape.tuple_shapes().empty()) return 1;
    return WorkItemsFromShape(shape.tuple_shapes(0));
  }
  if (!shape.IsArray() || shape.dimensions().empty()) return 1;
  if (!shape.has_layout()) {
    return shape.dimensions(0);  // best-effort pre-layout fallback
  }
  int64_t major_dim = LayoutUtil::Major(shape.layout(), 0);
  return shape.dimensions(major_dim);
}
```

- [ ] **Step 4: Run — PASS. Commit.**

---

### Task B5: Implement `FusedKflops` and `FusedWorkItems`

- [ ] **Step 1: Write failing tests**

```cpp
TEST_F(CpuFusionCostModelTest, FusedKflopsIsSumOfProducerAndConsumer) {}
TEST_F(CpuFusionCostModelTest, FusedWorkItemsIsConsumerWorkItems) {}
```

- [ ] **Step 2: Implement**

```cpp
int64_t CpuFusionCostModel::FusedKflops(const HloInstruction& producer,
                                         const HloInstruction& consumer) const {
  return Kflops(producer) + Kflops(consumer);
}

int64_t CpuFusionCostModel::FusedWorkItems(const HloInstruction& /*producer*/,
                                            const HloInstruction& consumer) const {
  return IndependentWorkItems(consumer);
}
```

- [ ] **Step 3: PASS + commit.**

---

### Task B6: Implement `EstimateRunTimes`

- [ ] **Step 1: Write failing tests**

```cpp
TEST_F(CpuFusionCostModelTest, EstimateRunTimesReturnsPositiveUnfusedAndFused) {
  // p0 = f32[1024] parameter(0); a = add(p0, p0); b = multiply(a, p0).
  // Both durations > 0. unfused > fused for this fusion.
}

TEST_F(CpuFusionCostModelTest,
       EstimateRunTimesShowsFusedLargerThanUnfusedForMultipleDuplications) {
  // producer with 5 consumers — `fused` should include 5x producer time.
}
```

- [ ] **Step 2: Implement**

```cpp
FusionCostModel::RunTimes CpuFusionCostModel::EstimateRunTimes(
    const HloInstruction* producer,
    absl::Span<HloInstruction* const> consumers) {
  auto time_for = [this](int64_t kflops, int64_t bytes) -> absl::Duration {
    double compute_s =
        (kflops * 1000.0) / device_info_.peak_kflops_per_second;
    double mem_s = bytes / device_info_.peak_bytes_per_second;
    return absl::Seconds(std::max(compute_s, mem_s));
  };

  int64_t p_kflops = Kflops(*producer);
  int64_t p_bytes =
      static_cast<int64_t>(hlo_cost_analysis_->bytes_accessed(*producer));

  absl::Duration unfused = time_for(p_kflops, p_bytes);
  absl::Duration fused;
  for (const HloInstruction* c : consumers) {
    int64_t c_kflops = Kflops(*c);
    int64_t c_bytes =
        static_cast<int64_t>(hlo_cost_analysis_->bytes_accessed(*c));
    unfused += time_for(c_kflops, c_bytes);
    fused += time_for(p_kflops + c_kflops, p_bytes + c_bytes);
  }
  return {unfused, fused};
}
```

- [ ] **Step 3: PASS + commit.**

---

### Task B7: Implement `WouldExplodeIrSize` via `CodeDuplicationTooHigh`

**Critical:** Wraps the **actual** CPU cap (`FusionNodeIndexingEvaluation::CodeDuplicationTooHigh`), not the 16 KB `kFusionThresholdBytes` (which is only used inside the kDot branch on CPU).

- [ ] **Step 1: Write failing tests**

```cpp
TEST_F(CpuFusionCostModelTest,
       WouldExplodeIrSizeMatchesCpuInstructionFusionOnHighDuplication) {
  // Construct a small computation with a producer reused by many consumers
  // such that CpuInstructionFusion would reject. Compare.
}

TEST_F(CpuFusionCostModelTest,
       WouldExplodeIrSizeFalseForNormalFusion) {
  // Single-consumer elementwise fusion. Not in the "too high" regime.
}

TEST_F(CpuFusionCostModelTest, WouldExplodeIrSizeTrueAtFiveReductionCap) {
  // Consumer already a fusion with 5 reductions; producer is a reduce.
}
```

- [ ] **Step 2: Implement**

```cpp
bool CpuFusionCostModel::WouldExplodeIrSize(const HloInstruction* producer,
                                             const HloInstruction* consumer) {
  // 1. CodeDuplicationTooHigh — mirrors cpu_instruction_fusion.cc:437–443.
  auto [it, inserted] = fusion_node_evaluations_.try_emplace(
      consumer, FusionNodeIndexingEvaluation(consumer));
  if (it->second.CodeDuplicationTooHigh(producer)) {
    return true;
  }
  // 2. 5-reduction cap — mirrors cpu_instruction_fusion.cc:399–410.
  if (consumer->opcode() == HloOpcode::kFusion &&
      producer->opcode() == HloOpcode::kReduce) {
    static constexpr int64_t kMaxReductionsInFusion = 5;
    int64_t n = absl::c_count_if(
        consumer->fused_instructions(), [](const HloInstruction* i) {
          return i->opcode() == HloOpcode::kReduce;
        });
    if (n > kMaxReductionsInFusion) return true;
  }
  return false;
}
```

**On invalidation of `fusion_node_evaluations_`:** the base pass calls `FusionCostModel::OnInstructionFused` after every fusion. Override it on CPU to erase the specific stale entry:

```cpp
// Add to CpuFusionCostModel:
void OnInstructionFused(HloInstruction* /*producer*/,
                        HloInstruction* consumer,
                        HloInstruction* /*fusion*/) override {
  // Precise: only the consumer's entry is stale (its fused computation
  // just grew). Clearing the whole map would be O(N²) over a fusion run.
  // The default `precise_evaluation_invalidation = true` in options is
  // the intended path. Full-clear mode exists only for diagnosis.
  if (precise_invalidation_) {
    fusion_node_evaluations_.erase(consumer);
  } else {
    fusion_node_evaluations_.clear();
  }
}
```

Add `bool precise_invalidation_` private field, initialized from `CpuPriorityFusionOptions::precise_evaluation_invalidation` in the ctor.

- [ ] **Step 3: PASS + commit.**

---

## Phase C — `CpuPriorityFusion` pass

### Task C1: Add the flag

- [ ] **Step 1: Edit `xla/xla.proto`**

Around line 327 (after `xla_cpu_use_xnnpack = 359`), add:

```proto
  // If true, XLA:CPU uses the shared priority fusion pass
  // (xla/hlo/transforms/priority_fusion.h) with a CPU cost
  // model instead of CpuInstructionFusion.
  optional bool xla_cpu_use_priority_fusion = 503;
```

Verify 503 is unused:
```bash
grep -n "= 503;" /home/user/xla/xla/xla.proto
```
Expected: one match (your new line).

- [ ] **Step 2: Build, commit**

```bash
bazel build //xla:xla_proto_cc
git add xla/xla.proto
git commit -m "Add xla_cpu_use_priority_fusion debug option"
```

---

### Task C2: Skeleton of `CpuPriorityFusion`

**Files:**
- Create: `xla/service/cpu/cpu_priority_fusion.h`
- Create: `xla/service/cpu/cpu_priority_fusion.cc`
- Modify: `xla/service/cpu/BUILD`

- [ ] **Step 1: Header**

```cpp
/* Copyright 2026 The OpenXLA Authors. [standard header] */

#ifndef XLA_SERVICE_CPU_CPU_PRIORITY_FUSION_H_
#define XLA_SERVICE_CPU_CPU_PRIORITY_FUSION_H_

#include <cstdint>
#include <memory>

#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/priority_fusion.h"
#include "xla/service/cpu/cpu_fusion_cost_model.h"
#include "xla/service/hlo_cost_analysis.h"

namespace xla::cpu {

struct CpuPriorityFusionOptions {
  // Guard thresholds.
  int64_t kflops_refuse_threshold = 10'000;
  int64_t min_work_items = 32;
  ComplexOpPenalties complex_op_penalties{};
  CpuDeviceInfo device_info{};

  // Compile-time controls (safety valves; defaults don't limit quality).
  int64_t max_fusions_per_computation = 0;  // 0 = unlimited
  bool precise_evaluation_invalidation = true;  // plumbed to CpuFusionCostModel
  tsl::thread::ThreadPool* thread_pool = nullptr;  // passed to PriorityFusion base ctor
  // NOTE: wall_time_budget deferred — requires base-pass deadline check
  // that's out of scope for this CL. Add in a follow-up.
};

class CpuPriorityFusion : public PriorityFusion {
 public:
  // Constructs its own HloCostAnalysis and CpuFusionCostModel,
  // then hands ownership to the base class.
  CpuPriorityFusion(const AliasInfo* alias_info,
                    HloCostAnalysis::ShapeSizeFunction shape_size,
                    CpuPriorityFusionOptions options);

 protected:
  FusionDecision BackendCanFuse(HloInstruction* producer,
                                HloInstruction* consumer) override;
  bool IsFusible(const HloInstruction& instruction) override;

 private:
  // Helper factory that builds the owning cost model. Used to satisfy
  // base-class constructor ownership: we need to construct the cost
  // model (and own the HloCostAnalysis it references) BEFORE calling
  // PriorityFusion's constructor.
  struct CostModelBundle {
    std::unique_ptr<HloCostAnalysis> hlo_cost_analysis;
    std::unique_ptr<CpuFusionCostModel> cost_model;
  };
  static CostModelBundle BuildCostModel(
      HloCostAnalysis::ShapeSizeFunction shape_size,
      const CpuPriorityFusionOptions& options);

  // Private delegating ctor. Invoked by the public ctor after
  // BuildCostModel returns. Lets the base class receive
  // bundle.cost_model (transfer of ownership) while this class
  // retains bundle.hlo_cost_analysis.
  CpuPriorityFusion(const AliasInfo* alias_info,
                    CpuPriorityFusionOptions options,
                    CostModelBundle bundle);

  CpuPriorityFusionOptions options_;
  // LIFETIME CONTRACT — do not reorder, do not add destructor logic to
  // CpuFusionCostModel that dereferences hlo_cost_analysis_.
  //
  //   Construction: `BuildCostModel` creates `HloCostAnalysis` and
  //   `CpuFusionCostModel` (which captures a raw pointer into the
  //   analysis). Both move through `CostModelBundle` into this class.
  //   `std::unique_ptr<T>::get()` is stable across moves, so the
  //   captured raw pointer is valid for the cost model's lifetime.
  //
  //   Destruction: reverse member-declaration order on derived class:
  //   `cpu_cost_model_` (non-owning) destroyed first — it does not free
  //   anything. `hlo_cost_analysis_` destroyed next. THEN base-class
  //   destruction runs, destroying `cost_model_` (base's unique_ptr
  //   holding our CpuFusionCostModel). At that point
  //   `CpuFusionCostModel::~CpuFusionCostModel` executes with
  //   `hlo_cost_analysis_` already freed. The default destructor is a
  //   no-op, so we're safe — but any future destructor logic must not
  //   dereference `hlo_cost_analysis_`.
  std::unique_ptr<HloCostAnalysis> hlo_cost_analysis_;
  // Non-owning pointer to the base class's cost model, cast to the
  // concrete type for access to CPU-only helpers (FusedKflops etc.).
  CpuFusionCostModel* cpu_cost_model_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_PRIORITY_FUSION_H_
```

- [ ] **Step 2: `.cc`**

```cpp
#include "xla/service/cpu/cpu_priority_fusion.h"

#include <utility>
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace xla::cpu {

// static
CpuPriorityFusion::CostModelBundle CpuPriorityFusion::BuildCostModel(
    HloCostAnalysis::ShapeSizeFunction shape_size,
    const CpuPriorityFusionOptions& options) {
  HloCostAnalysis::Options analysis_options;
  analysis_options.shape_size = std::move(shape_size);
  auto hlo_cost_analysis =
      std::make_unique<HloCostAnalysis>(std::move(analysis_options));
  auto cost_model = std::make_unique<CpuFusionCostModel>(
      hlo_cost_analysis.get(), options.device_info,
      options.complex_op_penalties,
      options.precise_evaluation_invalidation);
  return {std::move(hlo_cost_analysis), std::move(cost_model)};
}

CpuPriorityFusion::CpuPriorityFusion(
    const AliasInfo* alias_info,
    HloCostAnalysis::ShapeSizeFunction shape_size,
    CpuPriorityFusionOptions options)
    : CpuPriorityFusion(alias_info, options,
                        BuildCostModel(std::move(shape_size), options)) {}

// Private delegating ctor: lets us both move `bundle.cost_model` into
// the base AND retain `bundle.hlo_cost_analysis` on `this`.
CpuPriorityFusion::CpuPriorityFusion(const AliasInfo* alias_info,
                                      CpuPriorityFusionOptions options,
                                      CostModelBundle bundle)
    : PriorityFusion(options.thread_pool, alias_info,
                     std::move(bundle.cost_model)),
      options_(std::move(options)),
      hlo_cost_analysis_(std::move(bundle.hlo_cost_analysis)),
      cpu_cost_model_(static_cast<CpuFusionCostModel*>(cost_model())) {}

FusionDecision CpuPriorityFusion::BackendCanFuse(HloInstruction* /*producer*/,
                                                  HloInstruction* /*consumer*/) {
  return FusionDecision::Allow();  // Task C3 fills in.
}

bool CpuPriorityFusion::IsFusible(const HloInstruction& /*instr*/) {
  return true;  // Task C4 fills in the CPU allowlist.
}

}  // namespace xla::cpu
```

Add a private delegating ctor declaration to the header (`CpuPriorityFusion(const AliasInfo*, CpuPriorityFusionOptions, CostModelBundle)`).

- [ ] **Step 3: BUILD target**

```python
cc_library(
    name = "cpu_priority_fusion",
    srcs = ["cpu_priority_fusion.cc"],
    hdrs = ["cpu_priority_fusion.h"],
    deps = [
        ":cpu_fusion_cost_model",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms:priority_fusion",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:instruction_fusion",
    ],
)
```

- [ ] **Step 4: Build + commit**

```bash
bazel build //xla/service/cpu:cpu_priority_fusion
git add xla/service/cpu/cpu_priority_fusion.{h,cc} xla/service/cpu/BUILD
git commit -m "Add CpuPriorityFusion skeleton with owning cost-model construction"
```

---

### Task C3: Implement the kflops/work-items guard + concat-on-minor-dim hard block

- [ ] **Step 1: Write failing tests**

Create `xla/service/cpu/cpu_priority_fusion_test.cc`:

```cpp
#include "xla/service/cpu/cpu_priority_fusion.h"

#include <string>
#include <gtest/gtest.h>
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

class CpuPriorityFusionTest : public HloHardwareIndependentTestBase {
 protected:
  CpuPriorityFusionOptions DefaultOptions() {
    CpuPriorityFusionOptions opts;
    opts.device_info.num_threads = 32;
    return opts;
  }
  HloCostAnalysis::ShapeSizeFunction ShapeSize() {
    return [](const Shape& s) { return CpuExecutable::ShapeSizeBytes(s); };
  }
  AliasInfo alias_info_;
};

TEST_F(CpuPriorityFusionTest, GuardRefusesHighKflopsLowWorkItemsFusion) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      p0 = f32[4, 1024] parameter(0)
      p1 = f32[4, 1024] parameter(1)
      c = f32[8, 1024] concatenate(p0, p1), dimensions={0}
      ROOT a = f32[8, 1024] add(c, c)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(std::string(kHlo)));
  CpuPriorityFusion pass(&alias_info_, ShapeSize(), DefaultOptions());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  const HloInstruction* root = m->entry_computation()->root_instruction();
  // Concatenate must not have been fused.
  EXPECT_NE(root->opcode(), HloOpcode::kFusion);
}

TEST_F(CpuPriorityFusionTest, GuardAllowsWhenWorkItemsSufficient) {
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      p0 = f32[32, 1024] parameter(0)
      p1 = f32[32, 1024] parameter(1)
      c = f32[64, 1024] concatenate(p0, p1), dimensions={0}
      ROOT a = f32[64, 1024] add(c, c)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(std::string(kHlo)));
  CpuPriorityFusion pass(&alias_info_, ShapeSize(), DefaultOptions());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
}

TEST_F(CpuPriorityFusionTest, GuardAllowsWhenKflopsBelowThreshold) { /* tiny pad */ }
TEST_F(CpuPriorityFusionTest, GuardFiresForDynamicUpdateSlice) { /* DUS + low work items */ }
TEST_F(CpuPriorityFusionTest, GuardFiresForDynamicSlice) {}
TEST_F(CpuPriorityFusionTest, GuardFiresForPad) {}

TEST_F(CpuPriorityFusionTest, ConcatOnMinorDimIsHardBlocked) {
  // f32[128, 128] concat on dim 1 with ≥128 elements on that dim: hard
  // refuse regardless of other metrics.
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      p0 = f32[128, 64] parameter(0)
      p1 = f32[128, 64] parameter(1)
      c = f32[128, 128] concatenate(p0, p1), dimensions={1}
      ROOT a = f32[128, 128] add(c, c)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(std::string(kHlo)));
  CpuPriorityFusion pass(&alias_info_, ShapeSize(), DefaultOptions());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  // Hard block: concatenate not fused.
  EXPECT_NE(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
}

TEST_F(CpuPriorityFusionTest, BackendCanFuseSurvivesMissingLayout) {
  // Module parsed without layout assignment; BackendCanFuse must not
  // CHECK-fail via LayoutUtil::Minor.
}

}  // namespace
}  // namespace xla::cpu
```

Add BUILD test target (mirror `cpu_fusion_cost_model_test`'s deps plus `:cpu_priority_fusion`).

- [ ] **Step 2: Run — FAIL.**

```bash
bazel test //xla/service/cpu:cpu_priority_fusion_test \
  --test_filter="CpuPriorityFusionTest.*"
```

Expected: all fail.

- [ ] **Step 3: Implement `BackendCanFuse`**

```cpp
FusionDecision CpuPriorityFusion::BackendCanFuse(HloInstruction* producer,
                                                  HloInstruction* consumer) {
  // 1. Bitcast-consumer forbid. Moved here from the generic CanFuse
  //    so each backend owns its ordering. Matches priority_fusion.cc:803.
  if (IsFusibleBitcast(*consumer)) {
    return FusionDecision::Forbid(
        "not fusing into a single bitcast as consumer");
  }

  // 2. Concat-on-minor-dim hard block. Copied from
  //    cpu_instruction_fusion.cc:358–380, guarded by has_layout.
  auto is_minor_dim_concat = [](const HloInstruction* h) {
    if (h->opcode() != HloOpcode::kConcatenate) return false;
    if (h->shape().dimensions().size() <= 1) return false;
    if (!h->shape().has_layout()) return false;
    int64_t d = h->concatenate_dimension();
    return d == LayoutUtil::Minor(h->shape().layout(), 0) &&
           h->shape().dimensions(d) >= 128;
  };
  static constexpr int64_t kMaxConcatenateArguments = 8;
  auto concat_bad = [&](const HloInstruction* h) {
    return h->opcode() == HloOpcode::kConcatenate &&
           (h->operand_count() > kMaxConcatenateArguments ||
            is_minor_dim_concat(h));
  };
  if (concat_bad(producer) || concat_bad(consumer)) {
    return FusionDecision::Forbid("Concatenate fusion is inefficient.");
  }

  // 3. kflops / work-items guard.
  int64_t kflops = cpu_cost_model_->FusedKflops(*producer, *consumer);
  int64_t work_items = cpu_cost_model_->FusedWorkItems(*producer, *consumer);
  if (kflops > options_.kflops_refuse_threshold &&
      work_items < options_.min_work_items) {
    return FusionDecision::Forbid("kflops/work-items guard");
  }
  return FusionDecision::Allow();
}
```

- [ ] **Step 4: Run — PASS.**

```bash
bazel test //xla/service/cpu:cpu_priority_fusion_test
```

- [ ] **Step 5: Commit.**

---

### Task C4: Implement CPU-specific `IsFusible` override

- [ ] **Step 1: Write failing test**

```cpp
TEST_F(CpuPriorityFusionTest, IsFusibleRejectsScatterGatherReduceWindow) {
  // HLO containing gather — should not be fused.
  constexpr absl::string_view kHlo = R"(
    HloModule m
    ENTRY e {
      p0 = f32[100,3] parameter(0)
      idx = s32[2] parameter(1)
      g = f32[2,3] gather(p0, idx), offset_dims={1}, collapsed_slice_dims={0},
                   start_index_map={0}, index_vector_dim=1, slice_sizes={1,3}
      ROOT r = f32[2,3] add(g, g)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnVerifiedModule(std::string(kHlo)));
  CpuPriorityFusion pass(&alias_info_, ShapeSize(), DefaultOptions());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  // Gather must NOT appear inside any fusion in the module.
  for (const auto* computation : m->computations()) {
    for (const auto* instr : computation->instructions()) {
      EXPECT_NE(instr->opcode(), HloOpcode::kGather)
          << "gather should not have been fused";
    }
  }
  // (If gather appears outside a fusion, that's fine.)
}
```

- [ ] **Step 2: Implement**

```cpp
bool CpuPriorityFusion::IsFusible(const HloInstruction& instr) {
  // Match CpuInstructionFusion::CanBeLoopFused opcode allowlist.
  if (!instr.IsFusible()) return false;
  if (instr.opcode() == HloOpcode::kFusion) return true;

  // Non-scalar constant producers are rejected — mirrors
  // CpuInstructionFusion::ShouldFuse at cpu_instruction_fusion.cc:339-342.
  // Prevents 2-instruction constant-only fusions; scalar constants are
  // fine (often fused as broadcasts).
  if (instr.opcode() == HloOpcode::kConstant &&
      !ShapeUtil::IsEffectiveScalar(instr.shape())) {
    return false;
  }

  switch (instr.opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kIota:
    case HloOpcode::kPad:
    case HloOpcode::kReduce:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kCopy:
    case HloOpcode::kConstant:  // only scalar constants reach here
      return true;
    default:
      return instr.IsElementwise();
  }
  // Deliberately rejects kGather, kScatter, kReduceWindow vs GPU default.
}
```

- [ ] **Step 3: PASS + commit.**

---

### Task C5: Priority ordering sanity test

- [ ] **Step 1: Write test**

```cpp
TEST_F(CpuPriorityFusionTest, PrioritizesCheapElementwiseOverComplexOp) {
  // Cheap (add) and complex (pad) producers both fusible into a shared
  // consumer — the cheap one fuses first by priority. Assert via final
  // structure.
}
```

- [ ] **Step 2: Run. PASS (the `Priority` formula already gives the cheap path a better delta).**

- [ ] **Step 3: Commit.**

---

## Phase D — Pipeline integration

### Task D1: Wire the flag into `cpu_compiler.cc`

- [ ] **Step 1: Edit**

At `xla/service/cpu/cpu_compiler.cc:1050-1055`, replace:

```cpp
AliasInfo alias_info;
bool use_multi_output_fusion = options::UseMultiOutputFusion(module->config());
pipeline.AddPass<CpuInstructionFusion>(
    &alias_info,
    /*may_duplicate=*/!use_multi_output_fusion);
```

with:

```cpp
AliasInfo alias_info;
bool use_multi_output_fusion = options::UseMultiOutputFusion(module->config());
if (module->config().debug_options().xla_cpu_use_priority_fusion()) {
  pipeline.AddPass<CpuPriorityFusion>(
      &alias_info, ShapeSizeBytesFunction(),
      CpuPriorityFusionOptions{});
} else {
  pipeline.AddPass<CpuInstructionFusion>(
      &alias_info, /*may_duplicate=*/!use_multi_output_fusion);
}
```

Add `#include "xla/service/cpu/cpu_priority_fusion.h"` to the include block.

- [ ] **Step 2: Update `xla/service/cpu/BUILD`**

In the `cpu_compiler` `cc_library` target, add `":cpu_priority_fusion"` to `deps`.

- [ ] **Step 3: Build**

```bash
bazel build //xla/service/cpu:cpu_compiler
```

Expected: success.

- [ ] **Step 4: Commit**

```bash
git add xla/service/cpu/cpu_compiler.cc xla/service/cpu/BUILD
git commit -m "Wire CpuPriorityFusion behind xla_cpu_use_priority_fusion flag"
```

---

### Task D2: Smoke test that the flag routes correctly

Per v2 design: this is a smoke test, **not** byte-identical HloProto parity (that would be vacuous — flag-off is the same `CpuInstructionFusion` path unchanged).

- [ ] **Step 1: Add test**

```cpp
TEST_F(CpuPriorityFusionTest, FlagRoutesToPriorityFusionWhenEnabled) {
  // Build a DebugOptions with xla_cpu_use_priority_fusion=true.
  // Run the CPU pipeline up through fusion.
  // Assert the module's last fusion-producing pass was CpuPriorityFusion,
  // not CpuInstructionFusion. Use HloModuleMetadata or pass names.
}

TEST_F(CpuPriorityFusionTest, FlagRoutesToInstructionFusionWhenDisabled) {
  // Opposite — default path taken.
}
```

`HloModuleMetadata` exposes pass-name history via `HloPassMetadata::pass_name` (see `xla/hlo/ir/hlo_module_metadata.h`). The test reads back the last fusion-phase pass name and asserts it equals `"priority-fusion"` (or `"cpu-instruction-fusion"` for the disabled case).

- [ ] **Step 2: PASS + commit.**

---

## Phase E — End-to-end validation

### Task E1: Flag-off baseline (parity sanity)

- [ ] **Step 1:** Pick two HLOs already in the tree for CPU testing. Suggested:
  - A shape-stable elementwise-heavy one from `xla/service/cpu/tests/`.
  - A DUS-heavy one (grep for `dynamic-update-slice` under `xla/service/cpu/tests/`).

- [ ] **Step 2:** Run each through the CPU pipeline with `xla_cpu_use_priority_fusion=false` (default). Save post-fusion `HloProto` fingerprint as a reference string in the test.

- [ ] **Step 3:** Re-run with the flag explicitly `false`. Assert fingerprint equality. This guards against accidental wiring drift.

- [ ] **Step 4:** Commit.

### Task E2: Flag-on correctness

- [ ] **Step 1:** Compile the same two HLOs with `xla_cpu_use_priority_fusion=true`.

- [ ] **Step 2:** Assert (a) compilation succeeds, (b) numerical output equivalence to flag-off on a small synthetic input (rtol=1e-5, atol=1e-6).

- [ ] **Step 3:** Log fusion count delta (informational, not asserted).

- [ ] **Step 4:** Commit.

---

## Open items for the implementer

Explicit deferrals. Do not resolve inside this plan's scope.

1. **Relaxing CPU caps** (`CodeDuplicationTooHigh`, 5-reduction, concat-on-minor-dim, 8-operand concat). Follow-up CL.
2. **Vectorization-aware cost modeling.** Structure allows a pure `CpuFusionCostModel` internal change.
3. **Removing `CpuInstructionFusion`.** Follow-up after the flag flips.
4. **`FusionFitsInBudget` CPU analog** for LLVM IR parameter-count blowup on many-operand fusions. Audit before the flag flips.
5. **Output fusion on CPU.** Today `CpuInstructionFusion` actively uses `CanBeOutputFused` / `CanBeOutputFusedIntoSomeOperand` (`cpu_instruction_fusion.cc:344–353`). The flag-on path currently does not produce output fusions. Before flipping the default, benchmark HLO that triggers output fusion today to quantify regression. If significant, either (a) teach priority fusion to produce output fusions via `ChooseKind` + a new `BackendCanFuse` branch, or (b) run `CpuInstructionFusion` in output-fusion-only mode after `CpuPriorityFusion`.
6. **Tuning `X = 10'000 kflops` and `min_work_items = 32`.** First guesses; benchmarks retune.
7. **Compile-time regression**. Expect 5–20× slowdown in the fusion phase vs `CpuInstructionFusion`, approximately 5–15% end-to-end. `precise_evaluation_invalidation=true` is the key mitigation. Measure on real workloads during rollout; wire up `max_fusions_per_computation` as a safety valve if needed.
8. **Wall-clock compile budget** (`wall_time_budget`). Intentionally not in this CL. Adding it requires a deadline check inside the base pass's queue loop that's out of scope. Add in a follow-up when benchmarks show we need it.

## Spec coverage check

| Spec requirement | Task(s) |
|---|---|
| Move `PriorityFusion` + `PriorityFusionQueue` to `xla/hlo/transforms/` | A2 |
| Relocate GPU caches to `GpuFusionCostModel` | A3 |
| `FusionCostModel` 5-method interface | A1 |
| GPU zero behavior change | A2+A3+A4, A5 |
| Update 4 GPU call sites | A4 |
| `IsFusible` virtualized | A2 step 7, C4 |
| `CpuFusionCostModel` with Kflops/WorkItems/FusedKflops/FusedWorkItems | B1–B5 |
| `EstimateRunTimes` returns `{unfused, fused}` | B6 |
| `WouldExplodeIrSize` via `CodeDuplicationTooHigh` | B7 |
| Complex-op penalties (no minor-dim multiplier) | B3 |
| `IndependentWorkItems` = physical outer dim, layout-safe | B4 |
| `CpuPriorityFusion::BackendCanFuse` guard | C3 |
| Concat-on-minor-dim hard block preserved | C3 |
| `CpuPriorityFusion::IsFusible` matches `CanBeLoopFused` | C4 |
| Flag `xla_cpu_use_priority_fusion`, default off | C1, D1 |
| Pipeline wiring | D1 |
| Flag routing smoke test | D2 |
| Flag-off parity | E1 |
| Flag-on correctness | E2 |
| FakeCostModel-driven base-pass test | A5 |
| Layout-guarded concat check survives no-layout module | C3 |

## Changelog vs v1

- Phase A expanded from 5 tasks to 5 tasks *with substantially more work per task*: queue extraction, cache relocation, and 4-caller rename are now in A2+A3+A4. Committed as one atomic step.
- A new FakeCostModel base-pass test in A5.
- `CpuFusionCostModel` gained `OnInstructionFused` override for `fusion_node_evaluations_` invalidation.
- `CpuPriorityFusion` constructor uses a private delegating ctor to solve the ownership/ordering problem — no `set_cost_model` band-aid.
- `WouldExplodeIrSize` implementation switched from fake 16 KB to `CodeDuplicationTooHigh`.
- `IsFusible` is a virtual override on both GPU (default allowlist) and CPU (subset rejecting scatter/gather/reduce-window).
- `IndependentWorkItems` computed from physical outer dim with layout guard and tuple-uses-first-element.
- Test-filter syntax fixed to single glob.
- BUILD dep lists expanded with explicit `bazel query deps(...)` verification step.
- Flag-off "byte-identical" test reframed as routing smoke test in D2; E1 does actual parity.

## Changelog vs v2

v2 reviewer found 5 new blockers/majors introduced by v2's rewrite. v3 fixes:

- **v2 dropped `InstructionFusion::ShouldFuseInPlaceOp`** (correctness regression). v3's Task A2 Step 6 restores it as the final generic step of `CanFuse`.
- **v2's `GpuFusionCostModel::RunInitialAnalysis` had no caller.** v3 adds `Prepare(HloComputation*)` to the `FusionCostModel` interface (Task A1). Base pass calls it in Task A2 Step 8 before constructing the queue.
- **v2's CPU `OnInstructionFused` did `fusion_node_evaluations_.clear()`** (O(N²) compile-time). v3 uses `fusion_node_evaluations_.erase(consumer)` gated by `precise_evaluation_invalidation=true` in options.
- **v2 didn't retain `operands_to_removed_consumers_runtimes_`** in the queue. v3's Task A2 Step 5 explicitly keeps it and documents the incremental math.
- **v2 hoisted `IsFusibleBitcast` to the generic `CanFuse`** before backend checks, changing GPU semantics. v3 moves the bitcast check into each backend's `BackendCanFuse` (GPU after Triton, CPU as first backend check).
- **v2's `-Inf` priority for constants** did not match CPU's actual non-scalar-constant rejection. v3 adds an explicit check at the top of `CpuPriorityFusion::IsFusible`.
- **v2 called output fusion a non-issue.** v3 adds it to Open Items with concrete resolution paths.
- **v2 had a wrong "scan" opcode reference.** v3 removes "scan" from prose (rejected set is scatter/gather/reduce-window).
- **v2 left `tiled_run_time_data_cache_` reachable from the shared queue via `GetTiledRunTimeDataCached`.** v3's Task A3 folds both the cache and the helper fully into `GpuFusionCostModel` as private members.
- **v3 adds 4 compile-time knobs** to `CpuPriorityFusionOptions` (`max_fusions_per_computation`, `precise_evaluation_invalidation`, `wall_time_budget`, `thread_pool`) and documents expected compile-time impact.
- **v3 adds explicit lifetime-ordering contract** for `CpuPriorityFusion::hlo_cost_analysis_` — blocks a foot-gun where a future refactor could silently invalidate the base class's cost-model pointer.
- v3 drops the unused `StashBundle` helper from v2.
- v3 tightens the D2 pass-name-history test to use the real `HloPassMetadata::pass_name` API.
- v3 fixes the `FakeCostModel` in Task A5 to use non-negative durations throughout.

## Changelog vs v3

v3 reviewer found 4 plumbing blockers (options declared but not threaded) + 2 majors + 3 minors. v4 fixes:

- **#10 (BLOCKER): CPU `HloCostAnalysis` never `Accept()`-ed.** v4 overrides `CpuFusionCostModel::Prepare(HloComputation*)` to do `computation->Accept(hlo_cost_analysis_)`. The base pass already calls `Prepare` before queue construction (added in v3). Pointer type changed from `const HloCostAnalysis*` to `HloCostAnalysis*` to allow the Accept call.
- **#11 (BLOCKER): CPU `BackendCanFuse` missing bitcast forbid.** v4 adds `if (IsFusibleBitcast(*consumer)) return Forbid(...)` as step 1 of CPU's `BackendCanFuse`. Matches spec §CPU subclass step 3(1).
- **#12 (BLOCKER): `precise_invalidation_` not plumbed.** v4 extends `CpuFusionCostModel` ctor to take `bool precise_invalidation` as its 4th parameter; `BuildCostModel` passes `options.precise_evaluation_invalidation`. Default-initialization foot-gun eliminated.
- **#13 (BLOCKER): `thread_pool` hardcoded to `nullptr`.** v4's delegating ctor passes `options.thread_pool` to the base `PriorityFusion` ctor.
- **#13b: `wall_time_budget` dropped from this CL.** Marked deferred with an explicit note in the options struct and as item 8 in Open Items — requires base-pass changes out of scope.
- **#14 (MAJOR): incremental math shown as prose only.** v4 pastes the full rewritten `CalculateProducerPriority` body in Task A2 Step 5 with field-by-field `RunTimes` subtraction.
- **#15 (MAJOR): GPU `IsFusible` relies on silent base default.** v4 adds an explicit comment on `GpuPriorityFusion` documenting the convention and contract.
- **#16 (MINOR): lifetime comment incomplete.** v4 extends the `hlo_cost_analysis_` contract to cover destruction order explicitly and prohibit destructor logic that dereferences the member.
- **#18 (MINOR): header missing delegating-ctor decl.** v4 adds the private delegating ctor declaration directly to the header code block.
- **#19 (NIT): duplicate item numbering.** v4 renumbers Open Items linearly.

## Changelog vs v4

v4 reviewer confirmed all 4 v3 plumbing blockers + 2 majors landed cleanly. Found 2 compile-level blockers + 2 minors + 2 nits. v5 fixes:

- **#1 (BLOCKER): ctor-arity shear.** Widening `CpuFusionCostModel` to 4 args broke Phase B test construction sites. v5 adds `= true` default for `precise_invalidation` so tests keep compiling without edits.
- **#2 (BLOCKER): `IsFusibleBitcast` file-local.** Pre-existing, surfaced by v4's CPU bitcast-forbid fix (now both GPU and CPU subclasses call it from separate `.cc` files). v5 promotes its declaration to `xla/hlo/transforms/priority_fusion.h` (definition stays in the shared `.cc`).
- **#3 (MINOR): spec still said `const HloCostAnalysis*`.** Updated to `HloCostAnalysis*` (non-const) with rationale.
- **#4 (MINOR): spec Risks listed `wall_time_budget` as a current knob.** Updated to list only the 3 present fields; `wall_time_budget` explicitly deferred.
- **#5 (NIT): incremental math `cached_priority` placeholder.** Replaced with concrete `reverse_map_.at(producer)->first.first`.
- **#6 (NIT): GPU `BackendCanFuse` step numbering had two "3"s.** Renumbered 1–6.
- **#7 (NIT): `OnInstructionFused` call site unspecified.** Pinned down: inside `PriorityFusion::Fuse()` at the tail, near old `priority_fusion.cc:~892`.

## Changelog vs v5

- **Phase A split into Phase 0 + Phase A.** All decoupling happens in place at `xla/backends/gpu/transforms/` (Phase 0). File relocation is a separate mechanical phase (A).
- **Tasks A1–A5 renamed to 0.1–0.5** with content adjusted for in-place execution: no `git mv`, `namespace xla::gpu` stays until Phase A, includes point at the existing GPU location for the base-pass header.
- **Added new Phase A tasks A.1 and A.2** covering the actual file relocation and test relocation.
- **Removed the "Commit A2+A3+A4 together" atomic step.** Each Phase 0 task commits independently. Task 0.2 (queue refactor) commits together with Task 0.3's minimal stub, or can split if the stub is provided first — plan explicitly describes both paths.
- **Task 0.5 no longer moves the test.** It adds the FakeCostModel base-pass test in place; Phase A.2 handles the move.
- **Header and include-guard text updated** throughout Task 0.2 to reflect that the file stays at `xla/backends/gpu/transforms/priority_fusion.h` through all of Phase 0.
