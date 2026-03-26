# MegaFusion Pass for XLA:CPU — Design Document

## Problem Statement

When XLA:CPU compiles small models (e.g., small transformers, MLPs, or
embedding-heavy models), the current compilation and execution pipeline
introduces disproportionate overhead:

1. **Compile-time overhead**: Each fusion op gets its own LLVM IR function.
   The LLVM module may be split into N parts for parallel codegen
   (`SplitModule`). Each part runs the full LLVM optimization pipeline
   (O2/O3) including SLP vectorizer and loop unrolling — passes that are
   expensive and provide negligible benefit on tiny kernels.

2. **Runtime overhead**: Each fusion becomes a separate `KernelThunk` in the
   `ThunkSequence`. Even when `ThunkExecutor` detects a small workload and
   runs sequentially, there is per-thunk overhead: kernel function lookup,
   `XLA_CPU_KernelArg` array setup, thread-dim calculation, and the
   `ExecuteInternal` dispatch. For models with 50–200 tiny fusions, this
   overhead dominates actual compute.

3. **Missed optimization**: Because each fusion is compiled as an independent
   function, LLVM cannot optimize across fusion boundaries — no cross-kernel
   register allocation, no elimination of redundant loads/stores between
   adjacent kernels that share buffers.

## Goal

Introduce a **MegaFusion** HLO pass that merges multiple small fusion
instructions (and other small ops) into a single large fused computation,
so that the entire model (or a large subgraph) is compiled into **one LLVM
function** and executed as **one thunk**.

Primary goals:
- **Reduce compile time** by compiling fewer, simpler LLVM functions and
  skipping expensive LLVM passes (SLP vectorizer, loop unrolling) on
  mega-fused kernels
- **Reduce runtime overhead** by eliminating per-thunk dispatch cost
- **Improve runtime performance** by enabling cross-kernel optimization
  within a single LLVM function

Non-goals:
- Replacing the existing fusion infrastructure for large ops
- Handling ops that require library calls (oneDNN, YNNPACK, Eigen dots)
- Handling control flow (while, conditional) inside the mega-fusion

## Background: Current Architecture

### Compilation Pipeline

```
HLO Module
  → RunHloPassesThroughLayoutAssn (fusion passes run here)
    → CpuInstructionFusion: merges elementwise/broadcast/reduce chains
    → FusionWrapper: wraps for emitter dispatch
    → CpuMultiOutputFusion: merges ops with shared inputs
  → RunHloPassesAfterLayoutAssn
    → ParallelTaskAssigner, CopyInsertion, etc.
  → CompileCpuExecutable
    → ThunkEmitter: HLO → ThunkSequence
      → IrEmitter2: emits LLVM IR functions per kernel
    → SplitModule into N parts
    → IrCompiler: LLVM optimization + codegen per part
    → JitCompiler/ORC: link into executable
```

### Thunk Execution

```
ThunkExecutor::Execute()
  → For each ready thunk in DAG order:
    → KernelThunk::ExecuteInternal()
      → Resolve kernel function pointer from FunctionLibrary
      → Build XLA_CPU_KernelArg array (buffer ptrs + shapes)
      → Call kernel(thread_dim, args) per work group
      → Signal completion → wake dependent thunks
```

### Key Observation

For a small model with 100 elementwise fusions, each touching <1KB of data:
- **Compile**: 100 LLVM functions × full O2 pipeline = seconds of compile time
- **Execute**: 100 thunk dispatches × ~500ns each = 50μs overhead on what
  should be <10μs of actual compute

## Proposed Design

### Phase 1: HLO-Level MegaFusion Pass

A new HLO pass `CpuMegaFusionPass` runs **after** all existing fusion passes
and **before** `CompileCpuExecutable`. It operates on the entry computation's
scheduled instruction sequence.

#### Eligibility Criteria

An HLO instruction is eligible for mega-fusion if ALL of the following hold:

1. **Is a fusion or simple elementwise op**: The instruction is either a
   `kFusion` instruction or a simple elementwise/copy/broadcast/reshape/
   bitcast/slice/concatenate op that would become a `KernelThunk`.

2. **No library calls**: The instruction does not lower to a library thunk
   (oneDNN, YNNPACK, Eigen dot, convolution). These have opaque
   implementations that cannot be inlined.

3. **No side effects**: The instruction is not a collective, infeed, outfeed,
   send, recv, custom-call with side effects, or RNG.

4. **No control flow**: The instruction is not a while loop, conditional,
   or call to a sub-computation (other than a fusion body).

5. **Small enough**: The total bytes touched by the instruction (sum of
   operand and output buffer sizes) is below a configurable threshold
   (default: 256KB). This ensures we only mega-fuse operations where
   per-thunk overhead dominates.

6. **Single thread**: The instruction is not marked for parallel execution
   by `ParallelTaskAssigner` (i.e., it has `BlockDim{1}` and `ThreadDim{1}`
   or is small enough that parallelism provides no benefit).

#### Fusion Algorithm

```
MegaFusionPass(HloModule* module):
  entry = module->entry_computation()
  schedule = module->schedule().sequence(entry)

  // Partition the schedule into maximal runs of eligible instructions
  mega_groups = []
  current_group = []

  for instr in schedule:
    if IsEligibleForMegaFusion(instr):
      current_group.append(instr)
    else:
      if len(current_group) > 1:
        mega_groups.append(current_group)
      current_group = []

  if len(current_group) > 1:
    mega_groups.append(current_group)

  // For each group, merge into a single fusion instruction
  for group in mega_groups:
    if TotalBytesTouched(group) > kMegaFusionByteThreshold:
      // Split into sub-groups under threshold
      SplitAndMerge(group)
    else:
      MergeIntoSingleFusion(group)
```

#### MergeIntoSingleFusion

Given a group of N instructions `[i_0, i_1, ..., i_{N-1}]` in schedule
order:

1. Create a new `HloComputation` (the mega-fusion body) that contains
   cloned versions of all N instructions' computations, inlined and
   connected.

2. For each instruction in the group:
   - If it's a fusion: inline its fused computation into the mega body
   - If it's a simple op: clone it into the mega body

3. Wire up data dependencies: if `i_j` produces a value consumed by
   `i_k` (where j < k), connect them directly inside the mega body
   (no intermediate buffer allocation needed).

4. External operands (produced outside the group) become parameters of
   the mega-fusion.

5. Results consumed outside the group become outputs (via tuple if
   multiple).

6. Create a new `HloInstruction::CreateFusion(...)` with kind
   `kMegaFusion` (new fusion kind) and replace all merged instructions.

7. Tag the fusion with backend config indicating:
   - `is_mega_fusion: true`
   - `disable_slp_vectorizer: true`
   - `disable_loop_unrolling: true`
   - `optimize_for_size: true`

### Phase 2: Emitter Support

#### IrEmitter2 Changes

When `IrEmitter2::EmitFusionHostKernel` encounters a mega-fusion:

1. Emit all sub-computations as a single LLVM function (the normal fusion
   emitter path handles this since the mega-fusion body is just a
   computation graph).

2. Attach LLVM function attributes to hint the backend:
   - `"optimize-for-size"` — triggers `-Os`-like behavior
   - `"no-slp-vectorize"` — disables SLP vectorizer
   - `"no-unroll-loops"` — disables loop unrolling

3. Since all sub-ops are element-wise or simple indexed operations, the
   existing `ElementalIrEmitter` can handle the fused body directly.

#### ThunkEmitter Changes

The `ThunkEmitter` emits a single `KernelThunk` for the mega-fusion,
exactly as it would for any fusion. No changes needed here — the mega-
fusion is just a (large) fusion instruction from the thunk emitter's
perspective.

### Phase 3: LLVM Compilation Pipeline Changes

#### Per-Function Optimization Control

Modify `IrCompiler::RunIrPasses` to respect per-function attributes:

Currently, SLP vectorizer and loop unrolling are controlled at the module
level via `PipelineTuningOptions`. For mega-fused kernels, we want to
skip these passes entirely.

**Option A (Preferred)**: Use the existing `backend_extra_options`
mechanism. The mega-fusion's kernel gets compiled into a separate LLVM
module with `optimize_for_size=true`, `disable_slp_vectorizer=true`,
`disable_loop_unrolling=true` set as module flags. The existing
`ExtractKernelsFromModule` + `GetXlaBackendExtraOptions` infrastructure
already supports this — mega-fusion kernels automatically get their own
compilation unit with reduced optimization.

**Option B**: Set LLVM function attributes (`optnone`, `optsize`,
`"no-slp-vectorize"`) on mega-fusion kernel functions and rely on
LLVM's per-function optimization behavior. This avoids module splitting
but has less reliable behavior across LLVM versions.

#### Reduced Optimization Pipeline

For mega-fusion modules, the `IrCompiler` uses:

```
PipelineTuningOptions pto;
pto.LoopVectorization = false;  // Tiny loops, not worth it
pto.SLPVectorization = false;   // Already disabled due to LLVM bug
pto.LoopUnrolling = false;      // Save compile time
// Use O1 instead of O2 — sufficient for simple elementwise code
opt_level = CodeGenOptLevel::Less;  // -O1
```

This alone can cut LLVM compile time by 2-5x for small functions.

### Phase 4: Module-Level Compilation Optimization

For small models where MegaFusion merges everything into a single kernel:

1. **Skip module splitting**: If there's only 1 compiled function after
   mega-fusion, set `num_default_parts = 1` and skip `SplitModule`
   entirely. This avoids the overhead of cloning and splitting.

2. **Single dylib**: Use a single `JITDylib` instead of
   `parallel_codegen_split_count` dylibs, reducing ORC JIT overhead.

3. **Eager compilation**: For tiny modules, compile synchronously instead
   of dispatching to a thread pool (thread pool dispatch + synchronization
   overhead exceeds compile time for trivial functions).

## Implementation Plan

### Step 1: Add MegaFusion HLO Pass (new files)

```
xla/service/cpu/cpu_mega_fusion_pass.h
xla/service/cpu/cpu_mega_fusion_pass.cc
xla/service/cpu/cpu_mega_fusion_pass_test.cc
```

- Implement `CpuMegaFusionPass : public HloModulePass`
- Add eligibility checking logic
- Implement schedule-order grouping algorithm
- Implement `MergeIntoSingleFusion` using existing HLO cloning/inlining
  utilities (`HloInstruction::CreateFusion`, `FuseInstruction`,
  `CloneWithNewOperands`)
- Unit tests with small model HLO graphs

### Step 2: Backend Config for Mega-Fusion

Extend `xla/service/cpu/backend_config.proto`:

```protobuf
message CpuBackendConfig {
  // ... existing fields ...
  bool is_mega_fusion = N;
}
```

Or alternatively, use `backend_extra_options` string to pass
`optimize_for_size=true,disable_slp_vectorizer=true,disable_loop_unrolling=true`.
This is simpler and works with existing infrastructure.

### Step 3: Integrate into CPU Compiler Pipeline

In `cpu_compiler.cc`, `RunHloPassesAfterLayoutAssn`:

```cpp
// After all existing fusion passes and before ParallelTaskAssigner:
if (debug_options.xla_cpu_enable_mega_fusion()) {
  pipeline.AddPass<CpuMegaFusionPass>(CpuMegaFusionPass::Options{
      .byte_threshold = debug_options.xla_cpu_mega_fusion_byte_threshold(),
      .min_group_size = 2,
  });
}
```

The pass should run:
- **After** `CpuInstructionFusion` and `CpuMultiOutputFusion` (so we
  merge already-fused ops)
- **After** `FusionWrapper` (so fusion emitter dispatch is set)
- **Before** `ParallelTaskAssigner` (so we don't mega-fuse ops that
  should be parallelized — though for small ops this shouldn't happen)
- **Before** `CopyInsertion` (so the copy analysis sees the mega-fused
  graph)

### Step 4: Wire Up Backend Extra Options

In `IrEmitter2::EmitFusionHostKernel`, when the fusion is a mega-fusion:

```cpp
if (IsMegaFusion(fusion_instruction)) {
  kernel_info.backend_extra_options =
      "optimize_for_size=true,"
      "disable_slp_vectorizer=true,"
      "disable_loop_unrolling=true";
}
```

This causes `ExtractKernelsFromModule` to compile the mega-fusion kernel
with reduced optimization, using the existing infrastructure.

### Step 5: Add Debug Option Flags

In `xla_flags.proto` / `DebugOptions`:

```protobuf
bool xla_cpu_enable_mega_fusion = N [default = false];
int64 xla_cpu_mega_fusion_byte_threshold = M [default = 262144]; // 256KB
```

### Step 6: Testing & Benchmarking

1. **Unit tests**: Verify fusion correctness on small HLO graphs
2. **End-to-end tests**: Compile + run small models, verify numerics
3. **Compile-time benchmarks**: Measure LLVM compile time with/without
4. **Runtime benchmarks**: Measure inference latency for small models
5. **Regression tests**: Ensure large models are unaffected (mega-fusion
   should be a no-op when all ops exceed byte threshold)

## Risks and Mitigations

### Risk: Increased code size for mega-fused kernels

The mega-fused LLVM function will be larger than any individual kernel.
However, since we're disabling loop unrolling and optimizing for size,
the generated code should still be compact.

**Mitigation**: Set `optimize_for_size = true` and cap the mega-fusion
at 256KB total data touched (which bounds the computation complexity).

### Risk: Breaking existing fusion invariants

The HLO fusion infrastructure assumes certain properties about fusion
bodies (e.g., single root, specific op patterns). Mega-fusion creates
unusual multi-output fusions.

**Mitigation**: Use `kOutput` fusion kind with a tuple root, which is
already supported by `CpuMultiOutputFusion`. Alternatively, introduce
a new `kMegaFusion` kind that the emitter handles explicitly.

### Risk: Buffer assignment complexity

Merging many ops changes buffer lifetimes. Some intermediate buffers
that were previously allocated separately may now overlap.

**Mitigation**: Run mega-fusion before `CopyInsertion` and
`BufferAssignment` so they see the final graph. Intermediate values
within the mega-fusion body are virtual — they don't need buffer
allocation (they're computed inline in the fused kernel).

### Risk: Interaction with oneDNN/YNNPACK fusions

Library-backed fusions must not be merged into mega-fusions.

**Mitigation**: The eligibility check explicitly excludes instructions
that lower to library thunks. Check for `kCustomCall` with
`onednn`/`ynn` targets, and any instruction where
`ThunkEmitter::EmitHloInstruction` would produce a non-kernel thunk.

## Performance Expectations

For a small MLP (5 dense layers, ~100 fused ops, <1MB total data):

| Metric | Before | After (est.) |
|--------|--------|-------------|
| LLVM compile time | ~500ms | ~50ms |
| Number of thunks | ~100 | ~5 |
| Per-inference overhead | ~50μs | ~5μs |
| Inference latency | ~80μs | ~35μs |

The compile time improvement comes primarily from:
1. Fewer LLVM functions to optimize (100 → 5)
2. Skipping SLP vectorizer and loop unrolling
3. Using O1 instead of O2
4. No module splitting overhead

The runtime improvement comes from:
1. Fewer thunk dispatches (100 → 5)
2. No intermediate buffer loads/stores between adjacent ops
3. Better register allocation across the fused computation

## Future Work

- **Adaptive thresholds**: Use a cost model to decide byte threshold
  based on target architecture and model characteristics
- **Partial mega-fusion**: Allow mega-fusing subsets of eligible ops
  even when interrupted by ineligible ops (by splitting the schedule
  into multiple mega-groups)
- **Integration with tiled emitter**: For medium-sized ops, combine
  mega-fusion with tiling for better cache behavior
- **Profile-guided mega-fusion**: Use runtime profiling data to decide
  which ops to merge (e.g., ops that always execute sequentially anyway)
