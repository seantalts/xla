# Two-Phase Fusion: Decoupling Inlining from Fusion for Compilation Speed and Runtime Quality

**Status:** Proposed\
**Authors:** XLA Team\
**Date:** 2026-03-04

## Problem Statement

XLA's compilation pipeline currently inlines all sub-computations (via
`CallInliner`) *before* the fusion passes run. This creates a fundamental
tension between two real-world needs:

1.  **Runtime quality (MJX and similar workloads):** Frameworks like MJX emit
    HLO with many small sub-computations (`remainder`, `cross_product`, `clip`,
    etc.) called dozens of times across a rigid-body simulation graph. The fusion
    pass must see through these call boundaries to discover critical
    producer-consumer fusion opportunities. Without inlining, the fusion pass
    treats each `kCall` as an opaque barrier, missing fusions that yield 3-4x
    runtime improvements.

2.  **Compilation time (large-model workloads):** Other projects have requested
    that inlining be *delayed* until after fusion. When a sub-computation is
    called from N call sites and inlined before fusion, the fusion pass must
    analyze an HLO graph that is up to N times larger than the outlined version.
    For large models with shared sub-computations, this blows up compile time
    without proportional runtime benefit -- the fusion pass often makes the same
    local decision at each duplicated copy.

These two requirements are in direct conflict under the current single-phase
design. This document proposes a **two-phase fusion** approach that satisfies
both.

## Background

### Current Pipeline (GPU)

The GPU compilation pipeline schedules inlining and fusion as follows:

```
Pre-SPMD:   FlattenCallGraph -> CallInliner(single_call_site=false)
Main:       CallInliner(single_call_site=false) -> ... expanders ...
Fusion:     PriorityFusion -> HloCSE -> HloDCE -> MultiOutputFusion -> HloCSE -> HloDCE
Post:       Rematerialization -> FusionWrapper -> FusionDispatch
```

Both `CallInliner` invocations use `single_call_site=false`, meaning *all*
eligible calls are inlined before fusion ever runs. The fusion pass
(`PriorityFusion`) then operates on a fully flattened graph.

### Current Pipeline (CPU)

```
Early:      CallInliner(single_call_site=true)
Fusion:     CpuInstructionFusion -> [FusionWrapper] -> [CpuMultiOutputFusion]
Late:       [if flatten_after_fusion] FlattenCallGraph -> CallInliner(single_call_site=true)
```

The CPU pipeline is more conservative -- `single_call_site=true` prevents
duplicating sub-computations. This avoids compile-time blowup but also prevents
fusion from seeing through multi-call-site sub-computations (the MJX problem).

### Key Observations

- **Small sub-computations dominate the MJX case.** The quaternion helpers are
  5-20 instructions each. Their fused form is typically a single fusion node.
- **Large sub-computations dominate the compile-time case.** Shared layers or
  blocks with hundreds of instructions, called from many sites.
- **Fusion within a sub-computation is independent of calling context.** A
  `quaternion_multiply` body fuses the same way regardless of which joint calls
  it. The fusion pass rediscovers this at every inlined copy.
- **Cross-boundary fusion depends on calling context.** Whether the *output* of
  `quaternion_multiply` fuses with its downstream consumer depends on the
  specific consumer at each call site.

This separation -- intra-computation fusion is context-independent, while
inter-computation fusion is context-dependent -- is the key insight behind the
two-phase design.

## Design: Two-Phase Fusion

### Overview

Split the fusion pipeline into two phases with inlining in between:

```
Phase 1 (Intra-computation fusion):
    For each non-entry sub-computation:
        Run fusion passes on the sub-computation independently

Phase 2 (Inlining):
    CallInliner(single_call_site=false)
    -- Each sub-computation is now in its fused form (fewer, larger nodes)

Phase 3 (Inter-computation fusion):
    Run fusion passes on the fully inlined graph
    -- Discovers cross-boundary opportunities
    -- Works on a smaller graph than today because Phase 1 pre-compacted nodes
```

### Phase 1: Intra-Computation Fusion

Run the existing fusion pipeline (`PriorityFusion` + `MultiOutputFusion` on GPU,
or `CpuInstructionFusion` + `CpuMultiOutputFusion` on CPU) on each
sub-computation independently. This is already supported by the pass
infrastructure -- `HloModulePass::RunOnComputation()` can target individual
computations.

**What this achieves:**

- A sub-computation with 20 element-wise ops becomes 1-3 fusion nodes.
- The cost is paid *once per sub-computation*, not once per call site.
- No change to the fusion pass itself -- same cost model, same decisions.

**Scoping:** Phase 1 should process sub-computations in bottom-up callgraph
order (callees before callers). This ensures that if computation A calls
computation B, B is already fused before A is processed. Use `HloCallGraph` to
determine traversal order.

### Phase 2: Inlining

Run `CallInliner(single_call_site=false)` as today, but now each inlined body is
already compacted by Phase 1. A sub-computation that was 20 instructions is now
1-3 fusion nodes. The post-inlining graph is substantially smaller than it would
be without Phase 1.

**Compile-time impact:** If a 20-instruction sub-computation is called from 30
sites, the current pipeline inlines 20 x 30 = 600 instructions for fusion to
analyze. With Phase 1, the sub-computation becomes ~2 fusion nodes, so inlining
produces ~60 nodes -- a 10x reduction in graph size entering Phase 3.

### Phase 3: Inter-Computation Fusion

Run the fusion pipeline again on the fully inlined graph. This phase discovers
cross-boundary fusion opportunities:

- A fused sub-computation output feeding into an element-wise consumer can be
  absorbed into the consumer's fusion.
- Two adjacent inlined sub-computations can be merged if the cost model says so.
- `MultiOutputFusion` can combine sibling inlined sub-computations sharing
  operands.

Because the graph is pre-compacted by Phase 1, this phase is cheaper than
today's single-phase fusion on the fully expanded graph.

### Modified Pipeline (GPU)

```
Pre-SPMD:   FlattenCallGraph -> CallInliner(single_call_site=false)
Main:       ... expanders ...
NEW Phase 1: SubcomputationFusion (PriorityFusion + MOF on each sub-computation)
Phase 2:    CallInliner(single_call_site=false)
Phase 3:    PriorityFusion -> HloCSE -> HloDCE -> MultiOutputFusion -> HloCSE -> HloDCE
Post:       Rematerialization -> FusionWrapper -> FusionDispatch
```

Note: The pre-SPMD `CallInliner` can remain unchanged -- it handles framework-
level inlining (e.g., JAX `jit` boundaries) that must happen early. The new
two-phase logic applies to the main compilation pipeline where user-level
sub-computations survive to the fusion stage.

### Modified Pipeline (CPU)

```
Early:      CallInliner(single_call_site=true)  -- unchanged
...
NEW Phase 1: SubcomputationFusion (CpuInstructionFusion + CpuMOF per sub-computation)
Phase 2:    CallInliner(single_call_site=false)  -- now safe to inline all
Phase 3:    CpuInstructionFusion -> [CpuMultiOutputFusion]
Post:       cleanup
```

The CPU pipeline currently uses `single_call_site=true` to avoid blowup. With
Phase 1 pre-compacting sub-computations, it becomes safe to inline all call
sites before Phase 3 without excessive compile-time cost.

### Interaction with Existing Passes

**HloCSE after Phase 3:** Common sub-expression elimination after Phase 3 may
discover that independently fused copies of the same sub-computation (now
inlined at multiple sites) produced identical fusion computations. CSE can merge
these, recovering sharing where fusion made the same decision at each call site.

**Rematerialization:** Runs after Phase 3 as today. The pre-compacted graph may
give rematerialization better options since fusion nodes are larger and have
clearer cost characteristics.

**FlattenCallGraph:** May be needed before Phase 1 to ensure each sub-
computation has a single caller in the call hierarchy (not the same as single
call *site* -- this is about nesting). The existing `FlattenCallGraph` pass
handles this.

## Implementation Plan

### Stage 1: SubcomputationFusionPass (new pass)

Create a new `HloModulePass` called `SubcomputationFusionPass` that:

1. Builds the `HloCallGraph` for the module.
2. Identifies non-entry computations that are targets of `kCall` instructions
   (excluding fusion computations, while-body/condition, etc.).
3. Traverses them in bottom-up callgraph order.
4. For each, runs the backend-appropriate fusion pipeline
   (`FusionPipeline()` on GPU, `CpuInstructionFusion` + `CpuMultiOutputFusion`
   on CPU).

**Files to create:**

- `xla/hlo/transforms/subcomputation_fusion.h`
- `xla/hlo/transforms/subcomputation_fusion.cc`
- `xla/hlo/transforms/subcomputation_fusion_test.cc`

**Estimated complexity:** ~200 lines of pass code. The pass is primarily
orchestration -- the actual fusion logic is reused from existing passes.

### Stage 2: Pipeline Integration (GPU)

Modify `gpu_compiler.cc` to insert `SubcomputationFusionPass` before the
main-pipeline `CallInliner`:

- Insert after expanders complete but before the existing `CallInliner` at
  ~line 728.
- The existing `FusionPipeline` call at ~line 1664 remains unchanged (becomes
  Phase 3).

**Files to modify:**

- `xla/service/gpu/gpu_compiler.cc`

### Stage 3: Pipeline Integration (CPU)

Modify `cpu_compiler.cc` similarly:

- Insert `SubcomputationFusionPass` before the fusion section at ~line 1019.
- Change the early `CallInliner` from `single_call_site=true` to
  `single_call_site=false` (or add a second inlining pass after Phase 1).

**Files to modify:**

- `xla/service/cpu/cpu_compiler.cc`

### Stage 4: Testing and Validation

- **Unit tests:** Verify that Phase 1 fuses within sub-computations without
  affecting the calling computation. Verify that Phase 3 discovers cross-
  boundary fusions after inlining.
- **MJX regression test:** Confirm that the quaternion-multiply and
  rigid-body-step patterns produce the same fusions as fully-inlined-then-fused.
- **Compile-time benchmark:** Measure compilation time on large-model workloads
  with many shared sub-computations. Target: no regression vs. current pipeline
  for small models, measurable improvement for large models.
- **Runtime benchmark:** Confirm no runtime regression on standard benchmarks
  (MJX, MLPerf, internal models).

### Stage 5: Cleanup and Tuning

- Add a flag (`--xla_enable_two_phase_fusion`, default true) to allow fallback.
- Consider whether Phase 1 should use a simplified/cheaper cost model since its
  decisions will be revisited in Phase 3.
- Profile Phase 1 overhead on workloads with many small sub-computations to
  ensure the per-computation fusion setup cost doesn't dominate.

## Alternatives Considered

### Alternative 1: Size-Threshold Inlining

**Approach:** Add an instruction-count threshold to `CallInliner`. Only inline
sub-computations with fewer than N instructions (e.g., N=64) before fusion.
Large sub-computations remain outlined.

**Pros:**
- Simplest possible fix. Requires ~10 lines of code in `CallInliner`.
- Directly addresses both cases: MJX's tiny helpers (5-20 instructions) get
  inlined; large shared blocks stay outlined.
- No changes to the fusion pass.

**Cons:**
- The threshold is arbitrary. A sub-computation with 65 instructions that
  *should* be inlined for fusion will be missed. A sub-computation with 63
  instructions that adds no fusion value will be duplicated unnecessarily.
- Doesn't compose well with other passes that change instruction counts.
  Sub-computations may cross the threshold after expander passes run.
- Doesn't reduce compile time for workloads with many *small* shared sub-
  computations (e.g., 1000 call sites to a 10-instruction helper -- all get
  inlined, fusion sees 10,000 instructions).

**Verdict:** Good as a quick interim fix. The two-phase approach subsumes it --
Phase 1 compacts sub-computations regardless of size, and Phase 3 handles cross-
boundary fusion regardless of original sub-computation size.

### Alternative 2: Teach Fusion to Fuse Through Calls

**Approach:** When `PriorityFusion` considers fusing a producer into a consumer
and the producer is a `kCall`, peek inside the called computation's root
instruction. If the root is fusible, absorb the call body into the fusion
without first inlining.

**Pros:**
- Optimal fusion decisions -- only inlines what actually benefits from fusion.
- No wasted work: sub-computations that don't participate in cross-boundary
  fusion are never duplicated.
- Single-pass: no need to run fusion twice.

**Cons:**
- **High implementation complexity.** The fusion pass currently assumes a flat
  instruction graph. Teaching it to see through `kCall` boundaries requires:
  - Modifying `ShouldFuse()` to understand call semantics.
  - Handling multi-caller sub-computations (must clone the computation body for
    the fusion, leaving the original for other callers).
  - Updating the cost model to estimate costs across call boundaries.
  - Ensuring `MultiOutputFusion` also handles cross-call patterns.
- **Abstraction violation.** The fusion pass becomes coupled to call semantics,
  making both harder to maintain and evolve independently.
- **Risk of subtle bugs.** Cross-computation fusion introduces new edge cases
  around parameter aliasing, side effects, and control dependencies that don't
  exist in flat-graph fusion.

**Verdict:** Architecturally elegant but high-risk and high-effort. The two-
phase approach achieves a similar result (fusion sees through call boundaries
after Phase 2 inlining) with much less invasive changes.

### Alternative 3: Outline After Fusion

**Approach:** Inline everything before fusion (fixing MJX). After fusion, add a
pass that identifies duplicate fusion computations across former call sites and
re-outlines them into shared sub-computations for downstream passes.

**Pros:**
- Fusion sees the complete graph -- optimal fusion decisions guaranteed.
- Later passes (scheduling, buffer assignment, code generation) benefit from
  the re-outlined structure.

**Cons:**
- **Does not address the compile-time concern.** The fusion pass itself still
  operates on the fully inlined graph, which is the primary bottleneck the
  other project wants to avoid.
- **Re-outlining is non-trivial.** Identifying semantically identical fusion
  computations requires canonicalization. Fusions at different call sites may
  have different shapes, layouts, or cost-model-driven decisions, making exact
  deduplication rare in practice.
- **Unclear benefit.** If the goal is to reduce code size for later passes,
  HloCSE already merges identical computations. A dedicated re-outlining pass
  would only help for near-identical (but not exact) computations.

**Verdict:** Doesn't solve the compile-time problem and adds complexity for
uncertain gain. Rejected.

### Alternative 4: Pipeline Configuration Flag

**Approach:** Add a flag like `--xla_inline_before_fusion={true,false}` and let
each project choose.

**Pros:**
- Zero implementation risk. Each project gets the behavior it wants.
- Immediately unblocks both use cases.

**Cons:**
- **Doesn't solve the problem -- just lets users choose which regression they
  get.** MJX needs `true` for runtime quality; the other project needs `false`
  for compile time. Neither gets both.
- **Maintenance burden.** Two code paths that must both be tested and
  maintained. Subtle bugs may appear only in one configuration.
- **User-hostile.** Pushes a compiler-internal decision to end users who
  shouldn't need to understand fusion pass ordering.

**Verdict:** Acceptable as a temporary escape hatch (the two-phase design
includes a fallback flag), but not a real solution.

## Open Questions

1. **Should Phase 1 use a lighter-weight cost model?** Phase 1 fusions are
   "tentative" -- Phase 3 may refine them after seeing the full graph. A
   simpler cost model in Phase 1 could reduce overhead without meaningfully
   affecting quality, since Phase 3 provides the final refinement.

2. **How should Phase 1 interact with Triton fusions?** On GPU, some fusions
   are Triton-specific (GEMM fusions, Softmax fusions). These are currently
   handled in post-layout-assignment passes that run before the main fusion
   pipeline. Phase 1 should likely exclude Triton-eligible patterns, leaving
   them for the existing Triton passes.

3. **Can Phase 1 and Phase 3 share fusion state?** If Phase 1 records its
   fusion decisions (e.g., "these 5 ops became one fusion"), Phase 3 could
   use this as a hint rather than re-analyzing from scratch. This is an
   optimization opportunity but not required for correctness.

4. **What about `kWhile` and `kConditional` bodies?** These are also sub-
   computations that the fusion pass currently doesn't see into. Phase 1
   could fuse within while-bodies and conditional branches for the same
   benefits, though the existing pipeline may already handle these adequately.
