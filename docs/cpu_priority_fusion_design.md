# Sharing Priority Fusion Between XLA:GPU and XLA:CPU

Status: Draft design (parts 1–2 of 3 — motivation, current state, high-level shape, interfaces)
Owner: `claude/gpu-priority-fusion-shared-LWbBQ`
Last updated: 2026-04-20

## Motivation

XLA:GPU has a sophisticated priority-queue fusion pass
(`xla::gpu::PriorityFusion`) that uses a cost model to estimate the
runtime benefit of each (producer, consumer) fusion candidate, fuses the
highest-benefit pair, and incrementally re-prioritizes affected edges.

XLA:CPU today uses `CpuInstructionFusion`, a greedy opcode/size-based
heuristic with no runtime estimate and no priority queue. There is no
CPU-specific `HloCostAnalysis` subclass, and fusion decisions are made
locally without considering impact on parallelism.

Several pain points on CPU motivate this work:

- Fusing `dynamic-update-slice`, `dynamic-slice`, `concatenate`, and
  `pad` has historically triggered crashes or pathological LLVM
  compilation times (e.g. `TODO(b/419635451)` in
  `cpu_instruction_fusion.cc:397`).
- CPU codegen emits `scf.parallel` over an outer tiling derived from
  `outer_dimension_partitions`, but fusion today has no notion of how
  fusing a producer into a consumer affects the number of independent
  parallel work items available to the runtime.
- Greedy local decisions systematically over-fuse cheap-but-wide ops
  into narrow ops, collapsing parallelism.

This doc proposes moving `PriorityFusion` into a backend-agnostic
location and parameterizing it with (a) a pluggable cost model that
reports both kFLOPs and **independent work items**, and (b) a pluggable
backend legality policy that encodes the known-bad-ops list and the
work-items-preservation rule.

## Non-goals

- Replacing `CpuInstructionFusion` immediately. The new pass will be
  opt-in behind a flag (`xla_cpu_use_priority_fusion`) and run in
  parallel for a bake-in period.
- Redesigning the GPU cost model. GPU behavior must be byte-identical
  after the move; the refactor is purely structural on the GPU side.
- Building a full CPU performance model. The CPU cost model in this doc
  is deliberately simple (FLOP-rate + bandwidth + work-items) and
  tunable; a richer model is future work.

## Current state

### GPU priority fusion

- Pass: `xla/backends/gpu/transforms/priority_fusion.{h,cc}`, class
  `xla::gpu::PriorityFusion : HloModulePass`
  (`priority_fusion.h:82`).
- Constructor takes: `tsl::thread::ThreadPool*`,
  `const se::DeviceDescription&`, `const AliasInfo*`,
  `GpuHloCostAnalysis::Options`, `mlir::MLIRContext*`.
- Internal `PriorityFusionQueue` (`priority_fusion.cc:154`) owns a
  `std::map<(Priority, id), HloInstruction*>` ordered by
  `Priority = time_unfused - time_fused` (`priority_fusion.cc:155`).
- Cost model: `GpuHloCostAnalysis` +
  `GpuPerformanceModel::EstimateRunTimes(...)` returning
  `RunTimes{time_unfused, time_fused}` via
  `EstimateRunTimeData{flops, bytes_read, bytes_written, read_time,
  write_time, compute_time, exec_time}`
  (`gpu_performance_model_base.h:41`).
- Legality: `CanFuse()` at `priority_fusion.cc:772` walks ~10 checks
  including Triton support, scatter emission, reduction-into-reduction,
  parameter budget, and merged-IR-size.

### CPU fusion

- `CpuInstructionFusion` at `xla/service/cpu/cpu_instruction_fusion.cc`
  overrides `ShouldFuse()`/`ChooseKind()` on top of base
  `InstructionFusion`. Policy is greedy, size-based, opcode-based.
- No `CpuHloCostAnalysis` — only the base `HloCostAnalysis` with
  `flops`/`bytes_accessed`/`optimal_seconds`.
- Parallelism signal lives in `BackendConfig::outer_dimension_partitions`
  (`elemental_kernel_emitter.cc:72`) and in tiled emitters'
  `thread_tile_sizes` (`cpu_fusion_emitter.cc:135`). This is where
  "independent work items" comes from on CPU.

### Shared-transforms conventions

- Backend-agnostic HLO passes live in `xla/hlo/transforms/`, namespace
  `xla::`, depending only on `//xla/hlo/pass:hlo_pass`,
  `//xla/hlo/ir:hlo`, and the base `HloCostAnalysis`.
- Backend-specific passes live in `xla/backends/{cpu,gpu}/transforms/`,
  namespace `xla::{cpu,gpu}::`, and may depend on backend-specific
  analyses.
- Precedent for base/derived analysis: `xla::HloCostAnalysis` →
  `xla::gpu::GpuHloCostAnalysis`. Precedent for polymorphic backend
  plug-ins: `xla::cpu::LibraryMatcher`.

## Proposed design — high-level shape

1. **Shared pass** at `xla/hlo/transforms/fusion/priority_fusion.{h,cc}`
   in namespace `xla::` holds the queue mechanics, candidate
   enumeration, cache invalidation, and the main drive loop. It takes
   two injected interfaces and a `ComplexOpPolicy`.
2. **Backend cost model interface** at
   `xla/hlo/analysis/fusion_cost_model.h` in namespace `xla::` —
   abstract class returning `FusionEstimate` with runtime, kFLOPs, and
   independent-work-items.
3. **Backend legality policy interface** at the same header — abstract
   class for the `CanFuse`/`ChooseKind`/`IsExpensive` hooks that are
   currently inlined into the GPU pass.
4. **GPU wrapper** at `xla/backends/gpu/transforms/priority_fusion.{h,cc}`
   keeps the old public API (`xla::gpu::PriorityFusion`) and constructs
   a `GpuFusionCostModel` + `GpuFusionPolicy`, then forwards to the
   shared pass. Zero GPU behavior change.
5. **CPU wrapper** at
   `xla/backends/cpu/transforms/priority_fusion.{h,cc}` constructs a
   `CpuFusionCostModel` + `CpuFusionPolicy` and runs the shared pass.

## Interfaces

### `FusionCostModel`

```cpp
// xla/hlo/analysis/fusion_cost_model.h
namespace xla {

struct FusionEstimate {
  // Wall-clock predictions used to compute priority.
  absl::Duration time_unfused;
  absl::Duration time_fused;

  // Total compute of the fused kernel, in thousands of FLOPs.
  int64_t fused_kflops;

  // Number of independent units the fused kernel can dispatch in
  // parallel. On GPU this is thread-blocks or tiles. On CPU this is
  // the product of outer_dimension_partitions (or the natural outer-dim
  // work count when partitions are unset).
  int64_t fused_independent_work_items;

  // Same metric for the inputs before fusion. Used to detect
  // "parallelism collapse" (fused < min(producer, consumer)).
  int64_t producer_independent_work_items;
  int64_t consumer_independent_work_items;
};

class FusionCostModel {
 public:
  virtual ~FusionCostModel() = default;

  // Estimate the benefit of fusing `producer` into `consumer`. The
  // implementation is allowed to cache internally.
  virtual absl::StatusOr<FusionEstimate> EstimateFusion(
      const HloInstruction* producer, const HloInstruction* consumer) = 0;

  // Called by the shared pass whenever an instruction's shape/users
  // change. Mirrors GpuHloCostAnalysis::RevisitInstruction.
  virtual absl::Status RevisitInstruction(const HloInstruction* instr) = 0;

  // Called when an instruction is removed from the module.
  virtual void ForgetInstruction(const HloInstruction* instr) = 0;
};

}  // namespace xla
```

The shared pass computes `priority = time_unfused - time_fused` exactly
as today; backends control the duration estimate.

### `BackendFusionPolicy`

```cpp
// xla/hlo/analysis/fusion_cost_model.h
namespace xla {

class BackendFusionPolicy {
 public:
  virtual ~BackendFusionPolicy() = default;

  // Backend-specific legality. Applied on top of the shared legality
  // checks (root-crossing, fusibility, in-place, cycles, budget).
  virtual FusionDecision CanFuse(const HloInstruction* producer,
                                 const HloInstruction* consumer) = 0;

  // FusionKind for the resulting fusion (kLoop, kInput, kCustom, ...).
  // On CPU today this is always kLoop.
  virtual HloInstruction::FusionKind ChooseKind(
      const HloInstruction* producer, const HloInstruction* consumer) = 0;

  // Is this op "expensive" for the purpose of duplication checks?
  virtual bool IsExpensive(const HloInstruction& instr) = 0;
};

}  // namespace xla
```

### `ComplexOpPolicy` and the kFLOPs / work-items refusal rule

The user flagged four ops that have crashed or generated pathological
code on CPU when fused: `dynamic-update-slice`, `dynamic-slice`,
`concatenate`, `pad`.

We encode the observation as a `ComplexOpPolicy` configuration owned by
the shared pass and consulted inside the drive loop after the cost
model returns a `FusionEstimate`:

```cpp
// xla/hlo/transforms/fusion/priority_fusion.h
namespace xla {

struct ComplexOpPolicy {
  // Opcodes that are "complex" enough that fusing them may hurt
  // codegen. Default {} on GPU, {dynamic-update-slice, dynamic-slice,
  // concatenate, pad} on CPU.
  absl::flat_hash_set<HloOpcode> complex_ops;

  // Refuse a fusion if BOTH of these hold:
  //   (1) fused_kflops > max_kflops_when_collapsing_parallelism
  //   (2) fused_independent_work_items < min_independent_work_items
  //       AND max(producer_work_items, consumer_work_items)
  //           >= min_independent_work_items
  //
  // In words: "don't allow a complex, expensive op to collapse a
  // kernel below 32-way parallelism when either input had that much
  // parallelism to begin with."
  int64_t max_kflops_when_collapsing_parallelism = 1024;  // tuned per backend
  int64_t min_independent_work_items = 32;
};

}  // namespace xla
```

The rule fires only when at least one of `producer` or `consumer` is in
`complex_ops`. GPU defaults to an empty set, preserving today's
behavior. The `max_kflops_when_collapsing_parallelism` threshold `X` is
a tuning knob; `1024` kFLOPs (≈ 1 MFLOP of compute in the fused kernel)
is a reasonable starting point — large enough to catch "heavy kernel
gets serialized by a pad" but small enough that genuinely tiny ops can
still fuse for memory-traffic savings.

Pseudocode inside the shared pass, invoked after `CanFuse` succeeds and
before the candidate is enqueued:

```cpp
FusionEstimate est =
    TF_RETURN_IF_ERROR(cost_model->EstimateFusion(producer, consumer));

const bool involves_complex =
    policy_.complex_ops.contains(producer->opcode()) ||
    policy_.complex_ops.contains(consumer->opcode());
const int64_t before = std::max(est.producer_independent_work_items,
                                est.consumer_independent_work_items);

if (involves_complex
    && est.fused_kflops > policy_.max_kflops_when_collapsing_parallelism
    && est.fused_independent_work_items < policy_.min_independent_work_items
    && before >= policy_.min_independent_work_items) {
  return FusionDecision::Forbid(
      "complex op would collapse parallelism below threshold");
}
```

This is the principled refusal rule the user asked for: a single
condition, parameterized on the two numbers every backend cost model
must return.

### CPU cost model sketch

`xla/backends/cpu/transforms/cpu_fusion_cost_model.{h,cc}`:

```cpp
class CpuFusionCostModel : public FusionCostModel {
 public:
  CpuFusionCostModel(const se::DeviceDescription& cpu_device_info,
                     HloCostAnalysis::ShapeSizeFunction shape_size);

  absl::StatusOr<FusionEstimate> EstimateFusion(
      const HloInstruction* producer,
      const HloInstruction* consumer) override;
  absl::Status RevisitInstruction(const HloInstruction* instr) override;
  void ForgetInstruction(const HloInstruction* instr) override;

 private:
  int64_t IndependentWorkItems(const HloInstruction* instr) const;
  absl::Duration ComputeTime(int64_t flops) const;
  absl::Duration MemoryTime(int64_t bytes) const;

  HloCostAnalysis cost_analysis_;  // base class is sufficient on CPU
  double peak_flops_per_second_;
  double peak_bw_bytes_per_sec_;
  int vector_width_;               // e.g. 8 floats for AVX2
  int num_threads_;
};
```

**Runtime estimate.** `exec_time = max(compute_time, memory_time)` —
the same roofline model GPU uses, with CPU constants.
`compute_time = flops / (peak_flops_per_second * vector_width)`;
`memory_time = bytes_accessed / peak_bw_bytes_per_sec`. This is crude
but sufficient to rank fusion candidates; it is not a substitute for
profiling.

**Work items.** `IndependentWorkItems(instr)` returns, in order of
preference:

1. `Product(instr->backend_config().outer_dimension_partitions())` if
   set and non-empty.
2. Otherwise, product of the outermost dimensions of `instr->shape()`
   that are `>= vector_width_` — the natural outer-loop parallelism the
   tiled emitter would exploit.
3. `1` for scalars / single-element shapes.

For a fusion candidate we estimate
`fused_independent_work_items = IndependentWorkItems(consumer)`, which
captures the fact that fusion flattens the producer into the consumer's
iteration space. Producer and consumer values are computed pre-fusion
for the refusal rule.

**kFLOPs.** `fused_kflops =
(flops(producer) + flops(consumer) + 999) / 1000`, using the base
`HloCostAnalysis`.

### CPU legality policy

`CpuFusionPolicy` encodes roughly what `CpuInstructionFusion::ShouldFuse`
does today:

- Reject non-loop-fusible ops (carry over `CanBeLoopFused` predicate).
- Reject concat on minor dimension
  (`cpu_instruction_fusion.cc:362-380`).
- Reject >5 reductions per fusion (b/419635451).
- Reject large (>10 KB) constants.
- Reject when producer is expensive AND consumer reuses operand
  elements.
- **Do not** duplicate the complex-op refusal rule here — that lives in
  the shared `ComplexOpPolicy` so GPU can enable it in the future.

Part 3 (to follow) will cover pipeline integration for GPU and CPU, the
migration plan, risks and open questions, and the citations appendix.
