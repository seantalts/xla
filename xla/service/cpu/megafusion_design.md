# MegaFusion Pass for XLA:CPU — Design Document

## Problem Statement

When XLA:CPU compiles small models (e.g., MuJoCo/MJX robotics, small
transformers, MLPs), the compilation and execution pipeline introduces
disproportionate overhead:

1. **Compile-time overhead**: Each fusion op gets its own LLVM IR function.
   The LLVM module may be split into N parts for parallel codegen
   (`SplitModule`). Each part runs the full LLVM optimization pipeline
   including loop unrolling — passes that are expensive and provide
   negligible benefit on tiny kernels.

2. **Runtime overhead**: Each fusion becomes a separate `KernelThunk` in
   the `ThunkSequence`. Even when `ThunkExecutor` detects a small workload
   and runs sequentially, there is per-thunk overhead: kernel function
   lookup, `XLA_CPU_KernelArg` array setup, and `ExecuteInternal`
   dispatch. For models with 50–200 tiny fusions, this overhead dominates
   actual compute. The DAG construction itself is O(N²) in thunk count.

3. **Missed optimization**: Because each fusion is compiled as an
   independent function, LLVM cannot optimize across fusion boundaries —
   no cross-kernel register allocation, no elimination of redundant
   loads/stores between adjacent kernels that share buffers.

## Goal

Introduce a **MegaFusion** HLO pass that merges multiple small HLO
instructions into a single large fused computation, compiled into
**one LLVM function** and executed as **one thunk**.

Primary goals:
- **Reduce compile time** by compiling fewer LLVM functions and skipping
  expensive LLVM passes (loop unrolling) on mega-fused kernels
- **Reduce runtime overhead** by eliminating per-thunk dispatch cost
- **Improve runtime performance** by enabling cross-kernel optimization
  within a single LLVM function

Non-goals:
- Replacing the existing fusion infrastructure for large ops
- Handling ops that require library calls (oneDNN, YNNPACK, Eigen dots)
- Handling control flow (while, conditional) inside the mega-fusion
- Fusing dot operations (see separate tiny-dot design doc)

## Background: Current Architecture

### Compilation Pipeline

```
HLO Module
  → RunHloPassesAfterLayoutAssn
    → LibraryRewriter (oneDNN/YNNPACK fusions → kCustom)
    → CpuInstructionFusion: merges elementwise/broadcast/reduce chains
    → FusionWrapper: wraps unfused ops for MLIR emitter dispatch
    → CpuMultiOutputFusion: merges ops with shared inputs
    → ParallelTaskAssigner: outlines large ops for parallel execution
    → CopyInsertion
  → CompileCpuExecutable
    → ThunkEmitter: HLO → ThunkSequence
      → For kLoop fusions (MLIR path):
        ParallelFusionEmitter → EmitFusionKernel() → FusionCompiler
      → For kCustom fusions: oneDNN/YNN thunks (not fusible)
      → Fallback: IrEmitter2 (legacy path)
    → IrCompiler: LLVM optimization + codegen
    → JitCompiler/ORC: link into executable
```

### MLIR Fusion Emitter Pipeline

The new MLIR path (gated by `xla_cpu_use_fusion_emitters` +
`UseExperimentalLoopFusion`) processes `kLoop` fusions:

1. `ParallelFusionEmitter::AddFusion()` calls `EmitFusionKernel()`
2. `EmitFusionKernel()` routes to:
   - `EmitTiledFusionKernel()` (XTile dialect, vectorized) — if tiling analysis succeeds
   - `EmitLoopFusionKernel()` (generic scalar loops) — fallback
3. `FusionCompiler::Compile()` runs MLIR pass pipeline → LLVM IR:
   - Scalar path: `AddScalarOptimizationPasses` + `AddScalarLoweringPasses`
   - Tiled path: `AddTiledOptimizationPasses` + `AddTiledLoweringPasses`
   - Transfers `ExtraBackendOptionsAttr` → LLVM module flag `xla_backend_extra_options`
4. Resulting `LlvmKernelSource` goes through `IrCompiler` for LLVM-level
   optimization and machine code generation

### ThunkEmitter kCustom Dispatch Problem

The ThunkEmitter's `kCustom` fusion handling (`thunk_emitter.cc:427-447`)
requires `backend_config.fusion_config().kind()` to be a recognized
library fusion kind (oneDNN or YNN). Unknown kinds return `Internal`
error. **This means we cannot simply use `kCustom` for mega-fusions
without modifying the ThunkEmitter.**

### CpuInstructionFusion Behavior

CIF fuses producer ops into consumer ops with careful heuristics:
- Caps reductions at 5 per fusion (`kMaxReductionsInFusion`) because
  fused reductions blow up in `X86TargetLowering::PerformDAGCombine`
  after loop unrolling
- Tracks code duplication via `FusionNodeIndexingEvaluation`
- Refuses to fuse fusion-into-fusion ("producer is itself a fusion node")
- Fuses INTO existing kLoop fusions (`consumer->IsLoopFusion()` → allow)

These guards exist primarily to control LLVM compile time from loop
unrolling. They are unnecessary when loop unrolling is disabled.

### Key Observation

For the MuJoCo/MJX benchmark (`many_small_ops_benchmark_test.cc`):
- `BM_ManySmallSequentialOps`: 200 adds on f64[6,6] — all fusible
- `BM_ManySmallDots`: chains of 6x6 dots — NOT fusible (dots)
- `BM_MixedSmallOps`: slice+broadcast+add+dot — partially fusible
- Total execution 0.02-0.13ms, per-thunk overhead dominates

## Proposed Design

### Core Idea: MegaFusion Runs Before CpuInstructionFusion

MegaFusion operates on **raw, unfused HLO ops**. It runs early in the
pipeline, aggressively merging small eligible ops. `CpuInstructionFusion`
then runs on whatever MegaFusion leaves behind.

This is simpler and cleaner than running after CIF because:

1. **No fusion-body inlining.** Merging raw HLO ops is straightforward
   cloning and wiring. Merging existing fusions requires inlining both
   fusion computations, remapping parameters, handling shared
   intermediates.

2. **CIF's compile-time guards are irrelevant.** The
   `kMaxReductionsInFusion` cap exists to prevent LLVM compile blowup
   from loop unrolling. Mega-fusions disable loop unrolling.

3. **Less wasted work.** CIF's O(N²) producer-consumer analysis only
   runs on ops MegaFusion didn't touch.

### Fusion Kind: kCustom with "\_\_mega\_fusion" kind

We use `FusionKind::kCustom` with a new `FusionBackendConfig.kind`
string `"__mega_fusion"`. This requires a small change to
`ThunkEmitter::EmitHloInstruction` to recognize the new kind and route
to the MLIR fusion emitter path:

```cpp
// In ThunkEmitter, kCustom fusion dispatch:
if (backend_config.fusion_config().kind() == kMegaFusionKind) {
  return EmitFusionKernelThunk(instruction);  // same as kLoop path
}
```

Using `kCustom` gives automatic compatibility:

- **CpuInstructionFusion**: `ComputeInstructionsToSkip()` skips
  instructions inside custom fusions. `ShouldFuse()` returns
  "Not fusing: producer is itself a fusion node." No changes needed.

- **ParallelTaskAssigner**: Returns task count 1 for custom fusions
  (`parallel_task_assignment.cc:172-178`). Correct — mega-fused ops
  are too small for parallelism.

- **FusionWrapper**: Won't touch kCustom fusions (only wraps raw
  unfused ops).

### Pipeline Placement

```
RunHloPassesAfterLayoutAssn:
  ...
  LibraryRewriter (oneDNN/YNNPACK)     ← creates kCustom library fusions
  CpuMegaFusionPass                    ← NEW: merge small eligible raw ops
  CpuInstructionFusion                 ← handles remaining ops normally
  FusionWrapper                        ← wraps remaining unfused ops for MLIR
  CpuMultiOutputFusion
  ...
  ParallelTaskAssigner                 ← skips kCustom mega-fusions
  CopyInsertion
  ...
```

### Eligibility Criteria

An HLO instruction is eligible for mega-fusion if ALL hold:

1. **Fusible op kind**: elementwise, bitcast, broadcast, concatenate,
   dynamic-slice, dynamic-update-slice, gather, iota, pad, reduce,
   reshape, reverse, slice, transpose, or copy. (Mirrors
   `CanBeLoopFused()` + the `FusionWrapper::MustWrapInstruction` list.)

2. **Not already fused**: Not a `kFusion` instruction (library fusions
   from `LibraryRewriter` are already `kCustom` fusions).

3. **Not a library call**: Not dot, convolution, custom-call, FFT.

4. **No side effects**: Not collective, infeed, outfeed, send, recv, RNG.

5. **No control flow**: Not while, conditional, or call.

6. **Small enough**: Output shape byte size below configurable threshold
   (default: 256KB). Matches `ParallelTaskAssigner`'s
   `min_cost_per_thread = 256KB`.

### Fusion Algorithm

```
CpuMegaFusionPass::Run(HloModule* module):
  computation = module->entry_computation()
  instructions = computation->MakeInstructionPostOrder()

  mega_groups = []
  current_group = []
  group_bytes = 0

  for instr in instructions:
    if not IsEligibleForMegaFusion(instr):
      FlushGroup(current_group, &mega_groups)
      continue

    bytes = ShapeUtil::ByteSizeOfElements(instr->shape())
    if group_bytes + bytes > kMegaFusionByteThreshold:
      FlushGroup(current_group, &mega_groups)

    current_group.append(instr)
    group_bytes += bytes

  FlushGroup(current_group, &mega_groups)

  for group in mega_groups:
    if len(group) < kMinGroupSize:  // default: 2
      continue
    MergeIntoMegaFusion(computation, group)
```

### MergeIntoMegaFusion

Given a group of N instructions in post-order:

1. **Identify external inputs**: operands of group instructions not
   in the group → become fusion parameters.

2. **Identify external outputs**: group instructions used outside the
   group or that are the computation root → become fusion outputs.

3. **Build fusion body**: New `HloComputation`. Clone each group
   instruction, remap operands (group members → clones, external →
   parameters).

4. **Create root**: Single output → direct root. Multiple → tuple root.

5. **Create fusion instruction**:
   ```cpp
   auto* mega = computation->AddInstruction(
       HloInstruction::CreateFusion(
           output_shape,
           HloInstruction::FusionKind::kCustom,
           external_inputs,
           fusion_computation));
   BackendConfig config;
   config.mutable_fusion_config()->set_kind("__mega_fusion");
   mega->set_backend_config(config);
   ```

6. **Replace uses**: For each external output, replace uses with the
   mega-fusion output (via `GetTupleElement` if multi-output).

7. **Clean up**: Remove original instructions.

### Emission Through MLIR Pipeline

In `ThunkEmitter::EmitHloInstruction`, the kCustom mega-fusion dispatch
routes to `EmitFusionKernelThunk`:

```cpp
if (backend_config.fusion_config().kind() == kMegaFusionKind) {
  return EmitFusionKernelThunk(instruction);
}
```

`EmitFusionKernelThunk` then routes the mega-fusion through the MLIR
emission pipeline via `ParallelFusionEmitter::AddFusion()`, which calls
`EmitFusionKernel()` → `EmitLoopFusionKernel()` (or tiled if eligible)
→ `FusionCompiler::Compile()` → LLVM IR.

**Important**: The existing check in `EmitFusionKernelThunk` gates the
MLIR path on `fusion_kind == kLoop`. We need to also allow
`kCustom` mega-fusions through:

```cpp
if (hlo_module_config_.debug_options().xla_cpu_use_fusion_emitters() &&
    options::UseExperimentalLoopFusion(hlo_module_config_) &&
    (fusion->fusion_kind() == HloFusionInstruction::FusionKind::kLoop ||
     IsMegaFusion(fusion)) &&
    fusion->fused_expression_root()->opcode() != HloOpcode::kDot) {
  // MLIR path
}
```

### LLVM Compilation: Reduced Optimization

Mega-fusion kernels use `LlvmKernelOptions` in `BackendConfig` to
control LLVM optimization. The proto already has exactly the fields
we need:

```protobuf
message LlvmKernelOptions {
  bool slp_vectorizer_disabled = 1;
  bool disable_loop_unrolling = 2;
  bool optimize_for_size = 3;
}
```

**Two paths converge**: Both the MLIR path (`FusionCompiler` transfers
`ExtraBackendOptionsAttr` → LLVM module flag) and the legacy path
(`SetXlaCpuBackendOptions` reads `LlvmKernelOptions` → LLVM module
flag) produce the same `xla_backend_extra_options` LLVM metadata.
`IrCompiler` reads this metadata via `GetXlaBackendExtraOptions()` and
configures `PipelineTuningOptions`:

```
pto.LoopVectorization = false
pto.SLPVectorization = false   // already disabled due to LLVM bug
pto.LoopUnrolling = false
```

For mega-fusions going through the MLIR path, we set
`ExtraBackendOptionsAttr` on the MLIR module during
`EmitFusionKernel`. The `FusionCompiler` already propagates this
attribute to the LLVM module (fusion_compiler.cc:497-508).

### Module-Level Compilation Optimization

For small models where MegaFusion merges everything into one or two
kernels:

1. **Skip module splitting**: ≤2 compiled functions → `num_default_parts = 1`,
   skip `SplitModule`. Avoids cloning overhead.

2. **Single dylib**: One `JITDylib` instead of `parallel_codegen_split_count`,
   reducing ORC JIT overhead.

## Implementation Plan

### Step 1: Add MegaFusion HLO Pass

New files:
```
xla/service/cpu/cpu_mega_fusion_pass.h
xla/service/cpu/cpu_mega_fusion_pass.cc
xla/service/cpu/cpu_mega_fusion_pass_test.cc
```

- `CpuMegaFusionPass : public HloModulePass`
- Eligibility checking (mirrors `CanBeLoopFused()` + size check)
- Post-order grouping algorithm
- `MergeIntoMegaFusion` using `HloInstruction::CreateFusion(kCustom)`
- Set `FusionBackendConfig.kind = "__mega_fusion"`
- Unit tests with small HLO graphs

### Step 2: ThunkEmitter Changes

In `thunk_emitter.cc`, modify `EmitHloInstruction`'s kCustom dispatch
to recognize mega-fusions:

```cpp
static constexpr char kMegaFusionKind[] = "__mega_fusion";

// In kCustom fusion handling:
if (backend_config.fusion_config().kind() == kMegaFusionKind) {
  return EmitFusionKernelThunk(instruction);
}
```

In `EmitFusionKernelThunk`, relax the kLoop gate to also accept
mega-fusions for the MLIR emitter path.

### Step 3: MLIR Emission Support

In `EmitFusionKernel` (fusion_emitter.cc), ensure mega-fusions are
recognized. They contain only ops that the loop emitter already handles.
Set `ExtraBackendOptionsAttr` on the MLIR module with:
```
optimize_for_size=true
disable_slp_vectorizer=true
disable_loop_unrolling=true
```

### Step 4: Pipeline Integration

In `cpu_compiler.cc`, `RunHloPassesAfterLayoutAssn`, insert MegaFusion
between LibraryRewriter and CpuInstructionFusion:

```cpp
if (debug_options.xla_cpu_enable_mega_fusion()) {
  pipeline.AddPass<CpuMegaFusionPass>(CpuMegaFusionPass::Options{
      .byte_threshold = debug_options.xla_cpu_mega_fusion_byte_threshold(),
      .min_group_size = 2,
  });
}
```

### Step 5: Debug Option Flags

Add to `DebugOptions` proto:
```protobuf
bool xla_cpu_enable_mega_fusion = N [default = false];
int64 xla_cpu_mega_fusion_byte_threshold = M [default = 262144];  // 256KB
```

### Step 6: Testing & Benchmarking

1. Unit tests: verify fusion structure on small HLO graphs
2. End-to-end tests: compile + run, verify numerics
3. Compile-time benchmarks: LLVM compile time with/without
4. Runtime benchmarks: `many_small_ops_benchmark_test.cc` with mega-fusion
5. Regression tests: large models unaffected (byte threshold rejects all)

## MuJoCo/MJX Impact Analysis

Based on the benchmarks in `many_small_ops_benchmark_test.cc`:

| Benchmark | MegaFusion Impact |
|-----------|------------------|
| `BM_ManySmallSequentialOps` (200 adds, f64[6,6]) | **HIGH**: All ops fused into 1 thunk |
| `BM_ManySmallDots` (50 dots, f64[6,6]) | **NONE**: Dots excluded (see tiny-dot design) |
| `BM_MixedSmallOps` (slice+bcast+add+dot) | **LOW**: Small groups between dots |
| `BM_CustomCallChain` (FFI calls) | **NONE**: Custom calls excluded |
| `BM_InterleavedComputeAndCustomCalls` | **LOW**: 1-2 ops between FFI calls |

MegaFusion addresses the pure-elementwise overhead. The MuJoCo
bottleneck of tiny dot products requires a separate approach (see
`tiny_dot_fusion_design.md`).

## Risks and Mitigations

### Risk: Large fusion body overwhelms MLIR emitter

**Mitigation**: The byte threshold (256KB) bounds data size, which
correlates with computation complexity. The MLIR scalar loop emitter
generates straightforward nested loops — 100 adds is just 100
operations in one loop body. With loop unrolling disabled, LLVM won't
explode the IR.

### Risk: Multi-output fusion complexity

**Mitigation**: Multi-output fusion with tuple root is already supported
by `CpuMultiOutputFusion` and the MLIR fusion emitters.

### Risk: CIF absorbs ops into mega-fusions

CIF would normally fuse producers into `kLoop` consumers. Since we use
`kCustom`, CIF's `ComputeInstructionsToSkip` skips instructions inside
custom fusions, and `ShouldFuse` refuses to fuse into/out of fusion
nodes. No changes needed.

### Risk: Buffer assignment for intermediates

**Not a risk**: Intermediates inside the fusion body don't need buffer
allocation — they're computed inline. This is normal fusion behavior
and actually reduces total buffer count.

## Performance Expectations

For `BM_ManySmallSequentialOps` with 200 ops:

| Metric | Before | After (est.) |
|--------|--------|-------------|
| Number of thunks | ~200 | 1 |
| LLVM compile time | ~200 functions | 1 function, O1 |
| ThunkExecutor DAG | O(200²) | O(1) |
| Per-inference dispatch | ~100μs overhead | ~0.5μs overhead |

For large models: MegaFusion is a no-op. Byte threshold rejects
everything, CIF runs as before, zero impact.

## Future Work

- **Cross-barrier mega-fusion**: Create multiple mega-groups when
  ineligible ops split the schedule
- **Integration with tiled emitter**: Use XTile path for medium ops
  within mega-fusions
- **Profile-guided thresholds**: Tune byte threshold per-architecture
- **Tiny dot fusion**: Separate design doc for fusing small dots
  via elemental emission rather than Eigen (see `tiny_dot_fusion_design.md`)
