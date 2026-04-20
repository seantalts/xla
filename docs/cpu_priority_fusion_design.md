# Sharing Priority Fusion Between XLA:GPU and XLA:CPU

Status: Draft design (part 1 of 3 — motivation, current state, high-level shape)
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

Parts 2 and 3 (to follow) will cover:

- Part 2 — Interfaces: full `FusionCostModel` and `BackendFusionPolicy`
  signatures, `ComplexOpPolicy` and the kFLOPs / work-items refusal
  rule, CPU cost model implementation sketch, CPU legality policy.
- Part 3 — Rollout: pipeline integration for GPU and CPU, migration
  plan, risks and open questions, citations appendix.
