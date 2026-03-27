# Tiny Dot Fusion for XLA:CPU — Design Document

## Problem Statement

The MuJoCo/MJX robotics workload consists of many small matrix operations
on tensors sized for typical robots (29-36 DoF). The dominant pattern is
chains of 6x6 matrix multiplications (spatial inertia, kinematics chains,
composite rigid body algorithm).

Currently, XLA:CPU handles these tiny dots suboptimally:

1. **Each 6x6 dot is a separate thunk.** Even though the actual
   computation is ~216 FMAs (6×6×6), each dot goes through the full
   thunk dispatch path: kernel function lookup, argument array setup,
   execution.

2. **Dots cannot be fused with adjacent ops.** `CpuInstructionFusion`
   only fuses dots in very limited cases (output fusion of
   matrix-vector `dot + add` where one operand fits in L1). Chain-of-dot
   patterns like `dot(dot(A, B), C)` never fuse.

3. **The dot implementation strategy is wrong for tiny dots.** The
   decision logic in `GetNonBatchDotImplementationStrategy` routes
   6x6 dots to `kTiledLlvmIrGemm` (since they pass `IsAlignedGemm`
   and `CanEmitTiledLlvmIrGemm` with `small_gemm = true`). While
   this avoids Eigen, the tiled GEMM emitter still has overhead
   from its tiling/blocking structure that's unnecessary for matrices
   that fit entirely in registers.

4. **Dots break MegaFusion groups.** The MegaFusion pass (see
   `megafusion_design.md`) excludes dots from eligibility. In the
   MuJoCo pattern `slice → broadcast → add → dot → slice → ...`,
   the dots break the chain into tiny 2-3 op groups that aren't
   worth mega-fusing.

## Goal

Enable tiny dot products (where all operands fit in registers or L1
cache) to be:
1. **Fused with adjacent elementwise ops** into a single kernel
2. **Emitted as simple loop nests** without tiling/blocking overhead
3. **Included in MegaFusion groups** so entire kinematic chains
   compile to a single function

## Background: Current Dot Handling

### Dot Implementation Strategy (`dot_op_emitter.cc:168-215`)

```
GetNonBatchDotImplementationStrategy(dot_info):
  if matrix-vector product:
    return kTiledLlvmIrGemv

  if both dimensions ≤ 3:
    return kNaiveLlvmIr

  if IsAlignedGemm:
    if CanEmitTiledLlvmIrGemm:  // k≤128, (m≤32,n≤128) or (m≤128,n≤32)
      return kTiledLlvmIrGemm
    elif allow_runtime_calls:
      return kEigen

  return kNaiveLlvmIr
```

For a 6x6 × 6x6 dot: m=6, k=6, n=6. `CanEmitTiledLlvmIrGemm` returns
true (6≤128, 6≤32, 6≤128). So the strategy is `kTiledLlvmIrGemm`.

### ThunkEmitter Dot Path (`thunk_emitter.cc:1038-1077`)

```
EmitDotThunk(instruction):
  strategy = GetDotImplementationStrategy(...)
  switch (strategy):
    kNaiveLlvmIr / kTiledLlvmIrGemm / kTiledLlvmIrGemv:
      → DotKernelEmitter (emits LLVM IR directly, not MLIR)
      → separate kernel thunk
    kEigen:
      → DotThunk (Eigen library call)
      → or YnnDotThunk if YNNPACK enabled
```

The `DotKernelEmitter` generates a standalone kernel — it cannot be
fused with adjacent operations.

### CpuInstructionFusion Dot Handling

CIF has very limited dot fusion (`cpu_instruction_fusion.cc:445-473`):
- Only fuses into dot consumers (output fusion for `dot + add`)
- Only for matrix-vector products where one operand < 16KB
- Dot-as-producer never fuses (it's in the "expensive" list)

### Why Dots Can't Be Fused Today

The fundamental issue: dots go through `DotKernelEmitter` which produces
a standalone LLVM kernel. The MLIR fusion pipeline
(`EmitFusionKernelThunk` → `ParallelFusionEmitter`) explicitly excludes
fusions with dot roots:

```cpp
// thunk_emitter.cc:875
fusion->fused_expression_root()->opcode() != HloOpcode::kDot
```

And `FusionWrapper::MustWrapInstruction` returns `false` for dots:
```cpp
// fusion_wrapper.cc:109
case HloOpcode::kDot:
  return false;
```

## Proposed Design

### Phase 1: Elemental Dot Emission in MLIR

Add support for emitting tiny dots as simple triple-nested loops in
the MLIR fusion emitter pipeline. For a dot `C[i,j] = Σ_k A[i,k] * B[k,j]`:

```mlir
// For 6x6 × 6x6:
scf.for %i = 0 to 6 {
  scf.for %j = 0 to 6 {
    %acc = arith.constant 0.0 : f64
    %sum = scf.for %k = 0 to 6 iter_args(%s = %acc) -> f64 {
      %a = memref.load %A[%i, %k] : memref<6x6xf64>
      %b = memref.load %B[%k, %j] : memref<6x6xf64>
      %prod = arith.mulf %a, %b : f64
      %next = arith.addf %s, %prod : f64
      scf.yield %next : f64
    }
    memref.store %sum, %C[%i, %j] : memref<6x6xf64>
  }
}
```

This is intentionally naive — LLVM will optimize the inner loops
(unroll the k=6 loop, vectorize if profitable). For 6x6 matrices
this generates excellent code without any tiling infrastructure.

#### Implementation

Extend the MLIR elemental emission to handle `HloOpcode::kDot` for
small dots. The existing `emitters::SubgraphToMlirFunction` (used by
`EmitLoopFusionKernel`) would need a handler for dot ops.

Alternatively, add a `TinyDotToMlir` lowering that converts small
dots to `linalg.matmul` or `linalg.generic` ops, which the existing
MLIR pipeline already knows how to lower through linalg → loops → LLVM.

**Size gate**: Only emit elementally if ALL dimensions (m, k, n) are
below a threshold (e.g., 32). Above that, the tiled emitter or Eigen
is still better.

### Phase 2: Allow Dots in Fusion Bodies

#### FusionWrapper Changes

In `fusion_wrapper.cc`, allow tiny dots to be wrapped:

```cpp
case HloOpcode::kDot:
  if (IsTinyDot(instruction)) {
    return using_new_fusion_emitter_;
  }
  return false;
```

Where `IsTinyDot` checks that all dimensions ≤ 32 (or a configurable
threshold).

#### CpuInstructionFusion Changes

In `cpu_instruction_fusion.cc`, allow tiny dots to participate in
fusion:

```cpp
bool CanBeLoopFused(const HloInstruction& hlo) {
  // ... existing cases ...
  if (hlo.opcode() == HloOpcode::kDot && IsTinyDot(hlo)) {
    return true;
  }
  return false;
}
```

Also modify `IsExpensive` to return `false` for tiny dots:

```cpp
case HloOpcode::kDot:
  return !IsTinyDot(instruction);
```

#### MLIR Loop Emitter Changes

The `EmitLoopFusionKernel` path needs to handle dot ops inside the
fusion body. Two approaches:

**Option A**: Lower dot to `linalg.matmul` during the MLIR emission.
The `stablehlo.dot_general` → `linalg.matmul` lowering already exists
in the StableHLO-to-Linalg pipeline. We'd emit the dot as a
`stablehlo.dot_general` and let the existing passes lower it.

**Option B**: Emit the dot as an explicit loop nest in the elemental
emitter. This is more work but gives us direct control over the
generated code.

Option A is preferred — it reuses existing infrastructure.

#### ThunkEmitter Changes

Remove the `kDot` exclusion in the MLIR path check:

```cpp
// Before:
fusion->fused_expression_root()->opcode() != HloOpcode::kDot

// After:
!(fusion->fused_expression_root()->opcode() == HloOpcode::kDot &&
  !IsTinyDot(*fusion->fused_expression_root()))
```

This allows fusions with tiny-dot roots to go through the MLIR path.

### Phase 3: MegaFusion Integration

Once tiny dots can be fused and emitted through the MLIR pipeline,
extend MegaFusion eligibility to include them:

```cpp
bool IsEligibleForMegaFusion(const HloInstruction* instr) {
  // ... existing checks ...
  if (instr->opcode() == HloOpcode::kDot) {
    return IsTinyDot(*instr);
  }
  // ...
}
```

This enables the MuJoCo pattern
`slice → broadcast → add → dot → slice → broadcast → add → dot → ...`
to fuse into a **single mega-fusion**, eliminating all intermediate
thunk dispatch and buffer materialization.

### IsTinyDot Definition

```cpp
bool IsTinyDot(const HloInstruction& dot) {
  CHECK_EQ(dot.opcode(), HloOpcode::kDot);
  const DotDimensionNumbers& dnums = dot.dot_dimension_numbers();

  // Only simple 2D matmul (no batch dimensions)
  if (dnums.lhs_batch_dimensions_size() > 0) return false;

  // All dimensions must be small
  const Shape& lhs = dot.operand(0)->shape();
  const Shape& rhs = dot.operand(1)->shape();
  const Shape& out = dot.shape();

  static constexpr int64_t kTinyDotThreshold = 32;

  for (int64_t dim : lhs.dimensions()) {
    if (dim > kTinyDotThreshold) return false;
  }
  for (int64_t dim : rhs.dimensions()) {
    if (dim > kTinyDotThreshold) return false;
  }
  for (int64_t dim : out.dimensions()) {
    if (dim > kTinyDotThreshold) return false;
  }
  return true;
}
```

The threshold of 32 means the largest dot we'd fuse is 32×32 × 32×32
= 32,768 FMAs — well within L1 cache and small enough that a simple
loop nest is competitive with tiled GEMM.

## Implementation Plan

### Step 1: IsTinyDot utility

Add `IsTinyDot()` to `ir_emission_utils.h/cc` (or a new shared header).
Used by FusionWrapper, CIF, MegaFusion, and ThunkEmitter.

### Step 2: MLIR elemental emission for dots

Extend `emitters::SubgraphToMlirFunction` or add a handler in the
loop fusion emitter for `HloOpcode::kDot`. Emit as
`stablehlo.dot_general` and let existing lowering handle it.

Test: standalone tiny-dot fusion → verify correct LLVM IR generation.

### Step 3: FusionWrapper + CIF changes

- `FusionWrapper::MustWrapInstruction`: return true for tiny dots
- `CpuInstructionFusion::IsExpensive`: return false for tiny dots
- `CanBeLoopFused`: include tiny dots
- `EmitFusionKernelThunk`: relax the `!= kDot` exclusion

Test: `dot + add` fusion, `add + dot + add` chain fusion.

### Step 4: MegaFusion integration

Add tiny dots to MegaFusion eligibility. Test full MuJoCo-like chains.

### Step 5: Debug flags

```protobuf
bool xla_cpu_enable_tiny_dot_fusion = N [default = false];
int64 xla_cpu_tiny_dot_threshold = M [default = 32];
```

### Step 6: Benchmarking

Run `many_small_ops_benchmark_test.cc`:
- `BM_ManySmallDots`: should see major improvement (50 thunks → 1)
- `BM_MixedSmallOps`: should see improvement (dots no longer break chains)
- `BM_InterleavedComputeAndCustomCalls`: improvement limited by FFI calls

## Risks and Mitigations

### Risk: Naive dot emission slower than DotKernelEmitter for ~32×32

For dimensions near the threshold, the tiled GEMM emitter may produce
better code due to cache blocking.

**Mitigation**: Start with threshold = 16 and benchmark up to 32.
LLVM's loop vectorizer + unroller on a 16×16 triple loop produces
good code. We can also let the tiled emitter handle dots in the
16-32 range and only fuse dots ≤ 16.

### Risk: Dot inside fusion breaks MLIR emitter assumptions

The MLIR loop fusion emitter currently assumes all ops are
elementwise or simple indexed operations. A dot introduces a
reduction dimension that doesn't map to the output indexing.

**Mitigation**: Use Option A (stablehlo.dot_general → linalg lowering).
The linalg pipeline already handles matmul correctly. The fusion
emitter just needs to emit the dot as a stablehlo op and the
existing MLIR passes do the rest.

### Risk: Interaction with DotDecomposer

`DotDecomposer` runs early in the pipeline and may restructure dots.
Need to ensure tiny dots survive decomposition intact.

**Mitigation**: `DotDecomposer` only decomposes dots with multiple
contracting dimensions or specific batch patterns. Simple 2D matmuls
(the MuJoCo case) pass through unchanged.

### Risk: Performance regression for medium dots

If the threshold is too aggressive, medium-sized dots (e.g., 64×64)
that currently use optimized tiled GEMM could be fused with a
suboptimal naive implementation.

**Mitigation**: Conservative threshold (16-32) + flag gating. Only
fuse dots that are clearly in the "overhead dominates compute" regime.
A 32×32 dot is ~32K FMAs ≈ 16μs at 2 GFLOPS/s. Thunk dispatch
overhead is ~0.5μs. So even at 32, fusion saves ~3% — the real win
is from eliminating intermediate buffer materialization and enabling
cross-op register allocation.

## Performance Expectations

For `BM_ManySmallDots` (50 chained 6x6 dots):

| Metric | Before | After (est.) |
|--------|--------|-------------|
| Number of thunks | 51 | 1 |
| Intermediate buffers | 50 × f64[6,6] | 0 (register-allocated) |
| Per-inference overhead | ~25μs dispatch | ~0.5μs |
| Compute | ~50 × 216 FMA | ~50 × 216 FMA |
| LLVM opportunity | none | unroll+vectorize across chain |

For `BM_MixedSmallOps` (slice+bcast+add+dot per joint, 6 joints):

| Metric | Before | After (est.) |
|--------|--------|-------------|
| Number of thunks | ~24 | 1 (with MegaFusion) |
| Intermediate buffers | ~18 | 0 |

## Future Work

- **Batched tiny dots**: Extend to batch dimensions ≤ small threshold
  (e.g., batch of 4 × 6×6 dots — common in robotics for quaternion ops)
- **Tiny dot + transpose fusion**: Handle `dot(A, B^T)` patterns
  directly without materializing the transpose
- **Auto-tuning threshold**: Profile-guided selection of the tiny dot
  threshold per target architecture
- **Tiled tiny dots**: For the 16-32 range, consider a lightweight
  tiling strategy (2×2 register tiles) rather than fully naive loops
