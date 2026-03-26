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

Introduce a **MegaFusion** HLO pass that merges multiple small HLO
instructions into a single large fused computation, so that the entire
model (or a large subgraph) is compiled into **one LLVM function** and
executed as **one thunk**.

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
  → RunHloPassesAfterLayoutAssn
    → LibraryRewriter (oneDNN/YNNPACK fusion)
    → CpuInstructionFusion: merges elementwise/broadcast/reduce chains
    → FusionWrapper: wraps for emitter dispatch
    → CpuMultiOutputFusion: merges ops with shared inputs
    → ParallelTaskAssigner: outlines large ops for parallel execution
    → CopyInsertion
  → CompileCpuExecutable
    → ThunkEmitter: HLO → ThunkSequence
      → IrEmitter2: emits LLVM IR functions per kernel
    → SplitModule into N parts
    → IrCompiler: LLVM optimization + codegen per part
    → JitCompiler/ORC: link into executable
```

### CpuInstructionFusion Behavior

`CpuInstructionFusion` fuses producer ops into consumer ops using
careful heuristics:

- Only fuses ops from `CanBeLoopFused()`: elementwise, bitcast,
  broadcast, concatenate, gather, iota, pad, reduce, reshape, reverse,
  slice, transpose
- Caps reductions at 5 per fusion (`kMaxReductionsInFusion`) because
  fused reductions cause `X86TargetLowering::PerformDAGCombine` to
  spend minutes combining load ops after loop unrolling
- Tracks code duplication via `FusionNodeIndexingEvaluation` to prevent
  fusion blowup
- Avoids minor-dimension concatenation fusion (leads to branchy inner
  loops)
- Refuses to fuse fusion-into-fusion ("producer is itself a fusion node")

These guards exist primarily to control LLVM compile time. They are
unnecessary when loop unrolling and SLP vectorization are disabled.

### Thunk Execution

```
ThunkExecutor::Execute()
  → Build DAG from buffer uses (O(N²) in thunk count)
  → For each ready thunk in DAG order:
    → KernelThunk::ExecuteInternal()
      → Resolve kernel function pointer from FunctionLibrary
      → Build XLA_CPU_KernelArg array (buffer ptrs + shapes)
      → Call kernel(thread_dim, args) per work group
      → Signal completion → wake dependent thunks
```

Even the sequential fast path (triggered for ≤8 thunks or ≤512B buffers)
has per-thunk overhead: buffer address resolution, kernel dispatch,
AsyncValueRef bookkeeping.

### Key Observation

For a small model with 100 elementwise ops, each touching <1KB of data:
- **Compile**: 100 LLVM functions × full O2 pipeline = seconds of compile time
- **Execute**: 100 thunk dispatches × ~500ns each = 50μs overhead on what
  should be <10μs of actual compute

## Proposed Design

### Core Idea: MegaFusion Runs Before CpuInstructionFusion

MegaFusion operates on **raw, unfused HLO ops** — not on existing fusion
instructions. It runs early in the pipeline, aggressively grabbing small
eligible ops. `CpuInstructionFusion` then runs on whatever MegaFusion
leaves behind, applying its careful heuristics to the remaining (larger)
ops.

This is simpler and cleaner than running after CIF because:

1. **No fusion-body inlining.** Merging raw HLO ops is straightforward
   cloning and wiring. Merging two existing fusions would require
   inlining both fusion computations, remapping parameters, and handling
   shared intermediates — significantly more complex.

2. **CIF's compile-time guards are irrelevant.** The `kMaxReductionsInFusion`
   cap, the `CodeDuplicationTooHigh` check, the concat heuristics — these
   exist to prevent LLVM compile blowup from loop unrolling and DAG
   combine. Mega-fusions disable loop unrolling, so the root cause is gone.

3. **CIF naturally skips mega-fused ops.** CIF refuses to fuse into or
   out of existing fusion instructions. Mega-fusions show up as `kCustom`
   fusions, which CIF explicitly skips. Zero code changes needed in CIF.

4. **Less wasted work.** CIF's O(N²) producer-consumer analysis only
   runs on ops MegaFusion didn't touch — typically just the large ops
   that actually benefit from CIF's careful heuristics.

### Pipeline Placement

```
RunHloPassesAfterLayoutAssn:
  ...
  LibraryRewriter (oneDNN/YNNPACK)     ← creates kCustom fusions, runs first
  CpuMegaFusionPass                    ← NEW: grab small eligible raw ops
  CpuInstructionFusion                 ← handles remaining ops normally
  FusionWrapper
  CpuMultiOutputFusion
  ...
  ParallelTaskAssigner                 ← skips kCustom mega-fusions
  CopyInsertion
  ...
```

MegaFusion runs after `LibraryRewriter` so that oneDNN/YNNPACK ops are
already marked as custom fusions and will be excluded from mega-fusion
eligibility.

### Eligibility Criteria

An HLO instruction is eligible for mega-fusion if ALL of the following
hold:

1. **Fusible op kind**: The instruction is one of the ops that
   `CanBeLoopFused()` would accept — elementwise, bitcast, broadcast,
   concatenate, dynamic-slice, dynamic-update-slice, gather, iota, pad,
   reduce, reshape, reverse, slice, transpose — or a copy.

2. **Not already fused**: The instruction is not already inside a fusion
   (i.e., it's in the entry computation, not a fusion body). It is not
   a `kFusion` instruction itself (those were created by LibraryRewriter
   or earlier passes).

3. **No library calls**: The instruction is not a dot (Eigen), convolution,
   custom-call, FFT, or any op that lowers to a library thunk.

4. **No side effects**: The instruction is not a collective, infeed,
   outfeed, send, recv, RNG, or any op with side effects.

5. **No control flow**: The instruction is not while, conditional, or call.

6. **Small enough**: The output shape byte size is below a configurable
   threshold (default: 256KB). This matches `ParallelTaskAssigner`'s
   `min_cost_per_thread = 256KB` — ops below this threshold wouldn't
   benefit from parallelism anyway.

### Fusion Algorithm

```
CpuMegaFusionPass::Run(HloModule* module):
  computation = module->entry_computation()
  instructions = computation->MakeInstructionPostOrder()

  // Build groups: walk post-order, greedily merge eligible ops
  // that form connected subgraphs
  mega_groups = []
  current_group = []
  group_bytes = 0

  for instr in instructions:
    if not IsEligibleForMegaFusion(instr):
      FlushGroup(current_group, mega_groups)
      continue

    bytes = ShapeUtil::ByteSizeOfElements(instr->shape())

    // Check: does this instruction connect to the current group?
    // (i.e., uses or is used by something in the group)
    // Also check byte budget
    if group_bytes + bytes > kMegaFusionByteThreshold:
      FlushGroup(current_group, mega_groups)

    current_group.append(instr)
    group_bytes += bytes

  FlushGroup(current_group, mega_groups)

  // Merge each group into a single kCustom fusion
  for group in mega_groups:
    if len(group) < kMinGroupSize:   // default: 2
      continue
    MergeIntoMegaFusion(computation, group)
```

### MergeIntoMegaFusion

Given a group of N instructions `[i_0, i_1, ..., i_{N-1}]` in
post-order:

1. **Identify external inputs**: operands of group instructions that
   are NOT themselves in the group. These become parameters of the
   mega-fusion computation.

2. **Identify external outputs**: group instructions whose values are
   used by instructions outside the group, or that are the root of
   the entry computation. These become outputs of the mega-fusion.

3. **Build the fusion body**: Create a new `HloComputation`. For each
   instruction in the group (in post-order):
   - Clone it into the fusion body
   - Remap operands: if the operand is another group member, point to
     its clone; if it's external, point to the corresponding parameter

4. **Create root**: If there's a single external output, it's the root.
   If multiple, create a tuple as the root.

5. **Create the fusion instruction**:
   ```cpp
   auto* mega = computation->AddInstruction(
       HloInstruction::CreateFusion(
           output_shape,
           HloInstruction::FusionKind::kCustom,
           external_input_operands,
           fusion_computation));
   ```

6. **Replace uses**: For each external output, replace all uses of the
   original instruction with the corresponding element of the
   mega-fusion output (via `GetTupleElement` if multi-output).

7. **Set backend config**: Tag with `backend_extra_options`:
   `optimize_for_size=true,disable_slp_vectorizer=true,disable_loop_unrolling=true`

8. **Clean up**: Remove the original instructions from the computation.

### Why kCustom Fusion Kind

Using `FusionKind::kCustom` gives us automatic compatibility with the
rest of the pipeline:

- **CpuInstructionFusion**: Explicitly skips custom fusions in
  `ComputeInstructionsToSkip()` and refuses to fuse producers that are
  fusion nodes. No changes needed.

- **ParallelTaskAssigner**: Explicitly returns task count 1 for custom
  fusions (`parallel_task_assignment.cc` lines 172-178). This is correct —
  mega-fused ops are too small for parallelism overhead.

- **FusionWrapper**: Needs to recognize mega-fusions and route them to
  the appropriate emitter. We tag via backend config so the wrapper can
  check.

- **ThunkEmitter**: Emits a single `KernelThunk` for the mega-fusion,
  same as any other fusion. The kernel name includes "mega" for
  debuggability.

### LLVM Compilation: Reduced Optimization

Mega-fusion kernels are compiled with reduced LLVM optimization via the
existing `backend_extra_options` infrastructure:

1. **IrEmitter2** sets `backend_extra_options` on the kernel:
   ```
   optimize_for_size=true
   disable_slp_vectorizer=true
   disable_loop_unrolling=true
   ```

2. **ExtractKernelsFromModule** (in `cpu_compiler.cc`) automatically
   extracts kernels with non-default backend options into separate
   LLVM modules.

3. **IrCompiler** reads module flags via `GetXlaBackendExtraOptions()`
   and configures `PipelineTuningOptions` accordingly:
   ```
   pto.LoopVectorization = false
   pto.SLPVectorization = false
   pto.LoopUnrolling = false
   ```

4. The mega-fusion module compiles at **O1 instead of O2** — sufficient
   for simple elementwise code, significantly faster.

No new LLVM infrastructure needed. This all works today.

### Module-Level Compilation Optimization

For small models where MegaFusion merges everything into one or two
kernels:

1. **Skip module splitting**: If there are ≤2 compiled functions after
   mega-fusion, set `num_default_parts = 1` and skip `SplitModule`.
   Avoids cloning overhead.

2. **Single dylib**: Use one `JITDylib` instead of
   `parallel_codegen_split_count` dylibs, reducing ORC JIT overhead.

3. **Eager compilation**: For tiny modules, compile synchronously
   instead of dispatching to a thread pool (dispatch + sync overhead
   exceeds compile time for trivial functions).

## Implementation Plan

### Step 1: Add MegaFusion HLO Pass

New files:
```
xla/service/cpu/cpu_mega_fusion_pass.h
xla/service/cpu/cpu_mega_fusion_pass.cc
xla/service/cpu/cpu_mega_fusion_pass_test.cc
```

- `CpuMegaFusionPass : public HloModulePass`
- Eligibility checking (mirrors `CanBeLoopFused()` list plus size check)
- Post-order grouping algorithm
- `MergeIntoMegaFusion` using `HloInstruction::CreateFusion()` with
  `kCustom` kind
- Backend config tagging
- Unit tests: small HLO graphs, verify fusion structure

### Step 2: Integrate Into Pipeline

In `cpu_compiler.cc`, `RunHloPassesAfterLayoutAssn`, insert MegaFusion
between LibraryRewriter and CpuInstructionFusion:

```cpp
// After library rewriting, before standard fusion passes:
if (debug_options.xla_cpu_enable_mega_fusion()) {
  pipeline.AddPass<CpuMegaFusionPass>(CpuMegaFusionPass::Options{
      .byte_threshold = debug_options.xla_cpu_mega_fusion_byte_threshold(),
      .min_group_size = 2,
  });
}

// Existing fusion passes — operate on what MegaFusion left behind
pipeline.AddPass<CpuInstructionFusion>(...);
```

### Step 3: Emitter Support

In `FusionWrapper` or `ThunkEmitter`, recognize mega-fusions (check
backend config) and route to the standard elemental fusion emitter.
The fusion body is just normal HLO ops — the existing
`ElementalIrEmitter` handles it.

In `IrEmitter2`, set `backend_extra_options` on the emitted kernel
when the source fusion is a mega-fusion.

### Step 4: Debug Option Flags

Add to `DebugOptions` proto:

```protobuf
bool xla_cpu_enable_mega_fusion = N [default = false];
int64 xla_cpu_mega_fusion_byte_threshold = M [default = 262144];  // 256KB
```

### Step 5: Testing & Benchmarking

1. **Unit tests**: Verify fusion correctness on small HLO graphs
2. **End-to-end tests**: Compile + run small models, verify numerics
3. **Compile-time benchmarks**: Measure LLVM compile time with/without
4. **Runtime benchmarks**: Measure inference latency for small models
5. **Regression tests**: Ensure large models are unaffected (all ops
   above byte threshold → MegaFusion is a no-op → CIF runs as before)

## Risks and Mitigations

### Risk: Mega-fusion body too large for elemental emitter

The elemental emitter generates code by recursively walking the fusion
body. A mega-fusion with 100+ ops could produce a very large LLVM
function.

**Mitigation**: The byte threshold (256KB) bounds the data size, which
correlates with computation complexity. Also, with loop unrolling and
SLP vectorizer disabled, LLVM won't explode the IR further. The
`kMaxReductionsInFusion` problem specifically comes from loop unrolling
interacting with `X86TargetLowering::PerformDAGCombine` — without
unrolling, this doesn't happen.

### Risk: Multi-output fusion complexity

When a mega-fusion has multiple external outputs, the root is a tuple.
The emitter needs to handle multi-output fusions correctly.

**Mitigation**: Multi-output fusion is already supported by
`CpuMultiOutputFusion` and the fusion emitters. Use the same
infrastructure.

### Risk: Buffer assignment for intermediates

Intermediate values between ops that are now inside the mega-fusion
become internal to the fusion body — they don't need separate buffer
allocations.

**Mitigation**: This is the normal behavior for fusion. Buffer
assignment only allocates buffers for fusion inputs and outputs, not
for intermediate values inside the fusion body. This is actually a
benefit — fewer buffers to allocate.

### Risk: Interaction with oneDNN/YNNPACK fusions

Library-backed fusions must not be absorbed into mega-fusions.

**Mitigation**: LibraryRewriter runs before MegaFusion. It creates
`kCustom` fusions for library ops. MegaFusion's eligibility check
excludes `kFusion` instructions entirely — it only operates on raw
unfused HLO ops. Library fusions are invisible to MegaFusion.

### Risk: CpuInstructionFusion gets confused by mega-fusions

CIF might try to fuse ops into a mega-fusion or vice versa.

**Mitigation**: CIF already handles this correctly with zero changes:
- `ShouldFuse()` returns "Not fusing: producer is itself a fusion node"
  for any fusion producer (line 421 of `cpu_instruction_fusion.cc`)
- `ComputeInstructionsToSkip()` skips instructions inside custom
  fusions
- CIF simply ignores mega-fusions and fuses the remaining ops normally

## Performance Expectations

For a small MLP (5 dense layers, ~100 small ops, <1MB total data):

| Metric | Before | After (est.) |
|--------|--------|-------------|
| LLVM compile time | ~500ms | ~50ms |
| Number of thunks | ~100 | ~5 |
| Per-inference overhead | ~50μs | ~5μs |
| Inference latency | ~80μs | ~35μs |

The compile time improvement comes from:
1. Fewer LLVM functions to optimize (100 → ~5)
2. Mega-fusion kernels skip SLP vectorizer and loop unrolling
3. O1 instead of O2 for mega-fusion kernels
4. Fewer module splits / dylibs

The runtime improvement comes from:
1. Fewer thunk dispatches (100 → ~5)
2. No intermediate buffer loads/stores between adjacent ops
3. Better register allocation across the fused computation
4. Simpler ThunkExecutor DAG (fewer nodes → cheaper O(N²) construction)

For large models (all ops > 256KB), MegaFusion is a no-op — the byte
threshold check rejects everything, CIF runs exactly as before, zero
performance impact.

## Future Work

- **Adaptive thresholds**: Use a cost model to decide byte threshold
  based on target architecture and model characteristics
- **Cross-barrier mega-fusion**: Allow mega-fusing ops separated by
  ineligible ops, creating multiple mega-groups per computation
- **Integration with tiled emitter**: For medium-sized ops, combine
  mega-fusion with tiling for better cache behavior
- **Profile-guided mega-fusion**: Use runtime profiling data to decide
  which ops to merge (e.g., ops that always execute sequentially anyway)
- **Whole-model fusion**: For truly tiny models, consider fusing
  everything (including small dots via elemental emission) into a
  single kernel
