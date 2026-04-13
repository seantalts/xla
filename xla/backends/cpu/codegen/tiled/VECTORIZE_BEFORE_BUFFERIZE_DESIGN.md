# Design: Vectorize Before Bufferize in XLA:CPU Tiled Pipeline

## Summary

Reorder the XLA:CPU tiled emitter pipeline so that elementwise operations
are vectorized **before** bufferization, not after. This eliminates the
linalg round-trip for elementwise ops, removes unnecessary intermediate
allocations, gets implicit fusion for free through SSA value forwarding,
and deletes three passes that exist only to bridge the current ordering.

## Motivation

### 1. Unnecessary linalg round-trip for elementwise ops

The current pipeline lowers StableHLO elementwise ops to arith
(`StablehloLowerToArithPass`), wraps them in `linalg.generic`
(`ConvertElementwiseToLinalgPass`), fuses the generics
(`FuseElementwisePass`), bufferizes, and then vectorizes the linalg
bodies back into the same arith ops on vectors
(`LinalgElementwiseToVectorPass`):

```
stablehlo.add (tensor)
  → arith.addf (tensor)               # StablehloLowerToArithPass
    → linalg.generic { arith.addf }   # ConvertElementwiseToLinalgPass
      → linalg.generic (memref)       # bufferization
        → vector.transfer_read
          + arith.addf (vector)        # LinalgElementwiseToVectorPass
          + vector.transfer_write
```

The wrapping into `linalg.generic` and subsequent unwrapping adds no
semantic value — the arith ops that end up in the vectorized output are
exactly the arith ops that went into the `linalg.generic`. By vectorizing
arith-on-tensor ops directly, we cut out the linalg detour:

```
stablehlo.add (tensor)
  → arith.addf (tensor)               # StablehloLowerToArithPass (keep)
    → vector.transfer_read
      + arith.addf (vector)            # VectorizeElementwisePass (new)
      + vector.transfer_write
        → bufferize
```

### 2. Unnecessary intermediate allocations

The tiling emitter produces one `tensor.empty()` per intermediate
elementwise result. `FuseElementwisePass` merges single-use chains
into a single `linalg.generic`, but multi-use intermediates survive.
After bufferization each surviving `tensor.empty()` becomes a
`memref.alloc()`.

Vectorizing before bufferization means intermediate values live as
vector SSA values — there is no tensor to allocate. Multi-use
intermediates are handled naturally by SSA: the vector value is simply
used by multiple consumers.

### 3. Implicit fusion through SSA forwarding

In the current pipeline, each elementwise op becomes a separate
`linalg.generic`, and `FuseElementwisePass` is needed to merge them.
The fusion control function only fuses single-use producers
(`producer->hasOneUse()`), and long chains require up to 1000 greedy
rewrite iterations.

With vectorization before bufferization, chains of elementwise ops
become chains of vector SSA values. Canonicalization folds
`vector.transfer_read(vector.transfer_write(v, t, idx), idx)` → `v`,
eliminating intermediate tensors automatically. No explicit fusion pass
is needed, and multi-use intermediates work correctly without code
duplication concerns.

### 4. Convergence with the GPU pipeline

The GPU emitter operates on higher-level abstractions and lowers tensors
late. Moving CPU vectorization before bufferization follows the same
principle and opens the door to sharing more pass infrastructure.

## Current Pipeline (Tiled Path)

From `fusion_compiler.cc : AddTiledOptimizationPasses` (lines 344–379):

```
RegisterOptimizationPasses           # shared: simplify-arith, canon, CSE
LowerXTileEntryPass
StablehloTargetIndependentOptimization
StablehloLowerToArithPass            # shlo elementwise → arith on tensors
ShloToVectorPass                     # shlo dot/reduce/broadcast/iota → vector
Canonicalize
LowerVectorMultiReduction
TensorOpsToBufferizablePass          # tensor.bitcast → arith.bitcast
StablehloLowerToXtilePass            # shlo convert/compare → arith on tensors
StablehloLegalizeToLinalgPass        # remaining shlo → linalg
ConvertElementwise0DTensorToScalar   # rank-0 tensor arith → scalar arith
ConvertElementwiseToLinalgPass       # arith on tensor → linalg.generic     ← REMOVE
FuseElementwisePass                  # merge linalg.generics                ← REMOVE
--- AddBufferizationPasses ---
LinalgElementwiseToVectorPass        # linalg.generic → vector r/w + arith  ← REMOVE
FoldMemRefAliasOps
Canonicalize, CSE
```

## Proposed Pipeline

```
RegisterOptimizationPasses
LowerXTileEntryPass
StablehloTargetIndependentOptimization
StablehloLowerToArithPass            # shlo elementwise → arith on tensors  [KEEP]
ShloToVectorPass                     # dot/reduce/broadcast/iota → vector   [KEEP]
Canonicalize
LowerVectorMultiReduction
TensorOpsToBufferizablePass
StablehloLowerToXtilePass            # convert/compare → arith on tensors   [KEEP]
StablehloLegalizeToLinalgPass        # non-elementwise shlo → linalg        [KEEP]
ConvertElementwise0DTensorToScalar   # rank-0 tensor arith → scalar         [KEEP]
VectorizeElementwisePass             # arith/math on tensors → vector       [NEW]
Canonicalize                         # fold transfer_read/write chains      [NEW]
--- AddBufferizationPasses ---
FoldMemRefAliasOps
Canonicalize, CSE
```

### Passes removed

| Pass | Why it can go |
|---|---|
| `ConvertElementwiseToLinalgPass` | Elementwise ops are vectorized directly from arith-on-tensor; the linalg.generic wrapper is unnecessary. |
| `FuseElementwisePass` | Fusion is implicit: chained vector ops share SSA values, and `transfer_read ∘ transfer_write` folds eliminate intermediate tensors. |
| `LinalgElementwiseToVectorPass` | Vectorization now happens before bufferization; the post-bufferization linalg vectorizer is unnecessary. |

### Pass added

| Pass | Description |
|---|---|
| **`VectorizeElementwisePass`** | Matches any op with the `Elementwise` trait on ranked tensor types (rank > 0). For each: (1) `ReadTensorToVector` each operand, (2) clone the op with vector types, (3) `WriteVectorToTensor` the result. Uses the existing `lowering_utils.h` infrastructure that `ShloToVectorPass` already uses. |

### Passes kept unchanged

| Pass | Why kept |
|---|---|
| `StablehloLowerToArithPass` | Handles the stablehlo→arith mapping including bf16/f16 type promotion for div/rem and the `xla.guard_ub` attribute for division-by-zero. Keeping it avoids duplicating this logic in the vectorize pass. |
| `StablehloLowerToXtilePass` | Handles `stablehlo.convert` (FP8 routing, float-to-int saturation/clamping, bf16→f32 promotion) and `stablehlo.compare` (predicate mapping, unsigned handling). The resulting arith ops on tensors are picked up by `VectorizeElementwisePass`. |
| `ConvertElementwise0DTensorToScalar` | Converts rank-0 tensor ops to scalar ops before vectorization. `VectorizeElementwisePass` skips rank-0 tensors since they don't benefit from vectorization. |
| `ShloToVectorPass` | Handles dot, reduce, broadcast_in_dim, iota — unchanged. |

## Key Design Decisions

### Reuse arith lowering instead of going directly from StableHLO

An alternative design would remove `StablehloLowerToArithPass` and
vectorize directly from StableHLO elementwise ops to vector ops. This
was considered but rejected for Phase 1 because:

1. **Type promotion for bf16/f16 div/rem** is already implemented in
   `StablehloLowerToArithPass`. The promotion produces a chain of arith
   ops (`extf` → `divf` → `truncf`) that `VectorizeElementwisePass`
   vectorizes naturally through SSA forwarding.

2. **`StablehloLowerToXtilePass`** handles `convert` and `compare` with
   complex logic (FP8 conversions, float-to-int saturation with three
   `select` ops, unsigned integer handling). Absorbing this would make
   the new pass much more complex.

3. **The new pass is trivially generic.** It matches *any* op with the
   `Elementwise` trait on tensor types. It does not need to know about
   specific ops, element types, or promotion rules. This makes it simple
   to implement and maintain.

4. **The arith representation is not a round-trip.** The concern about
   "round-tripping through arith" was really about the linalg detour:
   arith→linalg→arith. By going arith→vector directly, the arith ops
   appear once (on tensors) and then once (on vectors) — no wrapping
   and unwrapping.

The direct StableHLO-to-vector approach remains an option for a future
phase if consolidating passes provides measurable benefit.

### No padding needed

The original design proposed padding tensors to multiples of the vector
width. This is unnecessary for two reasons:

1. **The tiled emitter already pads tile dimensions to powers of 2.**
   `tiled_fusion_emitter.cc` uses `llvm::PowerOf2Ceil()` on all tile
   sizes (line 74, 84). Tiles entering the pipeline already have
   power-of-2 dimensions.

2. **`ReadTensorToVector` reads the entire tile into a single MLIR
   vector.** Following the same pattern as `ShloToVectorPass`, there
   are no inner tiling loops and no remainder handling. A
   `tensor<16x8xf32>` becomes `vector<16x8xf32>`. MLIR's vector
   lowering passes (`ConvertVectorToLLVMPass`, `ConvertVectorToSCFPass`)
   handle decomposition into physical register widths.

### Chain fusion through canonicalization

Consider two chained elementwise ops:

```mlir
%c = arith.addf %a, %b : tensor<8x8xf32>
%d = arith.mulf %c, %e : tensor<8x8xf32>
```

After `VectorizeElementwisePass` vectorizes both (in greedy rewrite
order):

```mlir
// addf vectorized:
%va = vector.transfer_read %a[0,0] : tensor<8x8xf32> → vector<8x8xf32>
%vb = vector.transfer_read %b[0,0] : tensor<8x8xf32> → vector<8x8xf32>
%vc = arith.addf %va, %vb : vector<8x8xf32>
%c  = vector.transfer_write %vc, %empty_c[0,0] : tensor<8x8xf32>

// mulf vectorized:
%vc2 = vector.transfer_read %c[0,0] : tensor<8x8xf32> → vector<8x8xf32>
%ve  = vector.transfer_read %e[0,0] : tensor<8x8xf32> → vector<8x8xf32>
%vd  = arith.mulf %vc2, %ve : vector<8x8xf32>
%d   = vector.transfer_write %vd, %empty_d[0,0] : tensor<8x8xf32>
```

After canonicalization, the `transfer_read` of `%c` folds with the
`transfer_write` that defined it (`%vc2` → `%vc`), and the dead
`transfer_write` to `%c` is eliminated:

```mlir
%va = vector.transfer_read %a[0,0] : vector<8x8xf32>
%vb = vector.transfer_read %b[0,0] : vector<8x8xf32>
%vc = arith.addf %va, %vb : vector<8x8xf32>
%ve = vector.transfer_read %e[0,0] : vector<8x8xf32>
%vd = arith.mulf %vc, %ve : vector<8x8xf32>
%d  = vector.transfer_write %vd, %empty_d[0,0] : tensor<8x8xf32>
```

No intermediate allocation for `%c`. This works regardless of whether
the producer has one use or many — SSA semantics handle multi-use
naturally.

## Detailed Walkthrough: Elementwise Add

### Before (current pipeline)

```mlir
// Input (stablehlo, already tiled to 8x8):
%r = stablehlo.add %a, %b : tensor<8x8xf32>

// After StablehloLowerToArithPass:
%r = arith.addf %a, %b : tensor<8x8xf32>

// After ConvertElementwiseToLinalgPass:
%empty = tensor.empty() : tensor<8x8xf32>
%r = linalg.generic {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %s = arith.addf %lhs, %rhs : f32
    linalg.yield %s
} ins(%a, %b) outs(%empty) → tensor<8x8xf32>

// After bufferization:
%alloc = memref.alloc() : memref<8x8xf32>
linalg.generic { arith.addf } ins(%buf_a, %buf_b) outs(%alloc)

// After LinalgElementwiseToVectorPass (tiles to 8, loops over rows):
scf.for %i = 0 to 8 step 1 {
  %va = vector.transfer_read %buf_a[%i, 0] : vector<8xf32>
  %vb = vector.transfer_read %buf_b[%i, 0] : vector<8xf32>
  %vr = arith.addf %va, %vb : vector<8xf32>
  vector.transfer_write %vr, %alloc[%i, 0]
}
```

### After (proposed pipeline)

```mlir
// Input (stablehlo, already tiled to 8x8):
%r = stablehlo.add %a, %b : tensor<8x8xf32>

// After StablehloLowerToArithPass:
%r = arith.addf %a, %b : tensor<8x8xf32>

// After VectorizeElementwisePass:
%va = vector.transfer_read %a[0, 0] : tensor<8x8xf32> → vector<8x8xf32>
%vb = vector.transfer_read %b[0, 0] : tensor<8x8xf32> → vector<8x8xf32>
%vr = arith.addf %va, %vb : vector<8x8xf32>
%empty = tensor.empty() : tensor<8x8xf32>
%r = vector.transfer_write %vr, %empty[0, 0] : tensor<8x8xf32>

// After bufferization:
// Bufferization's alias analysis can often eliminate the empty tensor
// allocation when the result is written directly to an output buffer.
// The vector.transfer_read/write ops on tensors are natively supported
// by MLIR's one-shot bufferization.

// After vector lowering (ConvertVectorToSCFPass, ConvertVectorToLLVMPass):
// MLIR decomposes vector<8x8xf32> into hardware-width operations
// automatically. No manual tiling or peeling in our code.
```

### Chained ops with bf16 type promotion (div)

```mlir
// Input:
%r = stablehlo.div %a, %b : tensor<8x8xbf16>

// After StablehloLowerToArithPass (promotes to f32):
%ea = arith.extf %a : tensor<8x8xbf16> → tensor<8x8xf32>
%eb = arith.extf %b : tensor<8x8xbf16> → tensor<8x8xf32>
%div = arith.divf %ea, %eb : tensor<8x8xf32>
%r = arith.truncf %div : tensor<8x8xf32> → tensor<8x8xbf16>

// After VectorizeElementwisePass + canonicalization:
// Each arith op is vectorized independently; transfer_read/write folds
// eliminate all intermediate tensors:
%va = vector.transfer_read %a[0,0] : vector<8x8xbf16>
%vb = vector.transfer_read %b[0,0] : vector<8x8xbf16>
%vea = arith.extf %va : vector<8x8xbf16> → vector<8x8xf32>
%veb = arith.extf %vb : vector<8x8xbf16> → vector<8x8xf32>
%vdiv = arith.divf %vea, %veb : vector<8x8xf32>
%vr = arith.truncf %vdiv : vector<8x8xf32> → vector<8x8xbf16>
%r = vector.transfer_write %vr ... : tensor<8x8xbf16>
// No intermediate allocations for extf, divf, or truncf results.
```

## Implementation Plan

### Phase 1: VectorizeElementwisePass + pipeline reorder

**New file:** `xla/backends/cpu/codegen/tiled/transforms/vectorize_elementwise_pass.cc`

The pass is a single rewrite pattern:

```cpp
class VectorizeElementwisePattern : public mlir::RewritePattern {
  // Match: any op with Elementwise trait on ranked tensor types (rank > 0)
  // Action:
  //   1. ReadTensorToVector() each operand  (from lowering_utils.h)
  //   2. Clone the op with vector result types
  //   3. WriteVectorToTensor() the result   (from lowering_utils.h)
  //   4. Replace original op
};
```

Register in `passes.td`, add to BUILD.

**Pipeline change in `fusion_compiler.cc`:**

```diff
   pm.addPass(xtile::createConvertElementwise0DTensorToScalarPass());
-  pm.addPass(mlir::createConvertElementwiseToLinalgPass());
-  pm.addPass(CreateFuseElementwisePass());
-  AddBufferizationPasses(pm);
-  pm.addPass(CreateLinalgElementwiseToVectorPass());
+  pm.addPass(CreateVectorizeElementwisePass());
+  pm.addPass(mlir::createCanonicalizerPass());
+  AddBufferizationPasses(pm);
```

Also remove the unused `stablehlo_to_linalg_options` variable (lines
360–362) — the options are set but never passed to the pass constructor.

**Testing:**
1. FileCheck test: `transforms/tests/vectorize_elementwise.mlir`
   - Single op (addf, mulf, etc.) on various shapes
   - Chain of two ops → verify intermediate is folded after
     canonicalization
   - Rank-0 tensors are skipped (left as scalar ops)
   - Mixed vector + arith-on-tensor IR (from ShloToVectorPass output)
2. Run existing XLA:CPU integration tests end-to-end
3. Benchmark allocation counts for multi-op elementwise fusions

### Phase 2: Cleanup

1. Delete `linalg_elementwise_to_vector_pass.cc` and its test if no
   other pipeline uses it.
2. Delete `fuse_elementwise_pass.cc` and its test if no other pipeline
   uses it.
3. Remove BUILD targets for deleted files.

### Phase 3: Optional consolidation

- Consider absorbing `StablehloLowerToArithPass` into
  `VectorizeElementwisePass` if profiling shows the extra pass overhead
  matters. This would go directly from stablehlo elementwise ops to
  vector ops, eliminating the arith-on-tensor intermediate
  representation.
- Consider merging `VectorizeElementwisePass` into `ShloToVectorPass`
  for a single "tensor → vector" pass.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| One-shot bufferize may not handle vector-on-tensor IR well | MLIR's bufferization natively supports `vector.transfer_read/write` on tensors. `ShloToVectorPass` already produces this IR for dot/reduce/broadcast/iota and it bufferizes correctly today. |
| `transfer_read ∘ transfer_write` fold may not fire in all cases | The fold requires matching indices and the write covering the read region. Since we always use `[0, 0, ...]` indices and the vector shape matches the tensor shape, the fold is guaranteed. Add a canonicalization pass after `VectorizeElementwisePass` to ensure it fires before bufferization. |
| `StablehloLegalizeToLinalgPass` may try to lower ops we already lowered to arith | `StablehloLowerToArithPass` and `StablehloLowerToXtilePass` run first; by the time `StablehloLegalizeToLinalgPass` runs, all elementwise/convert/compare ops are already arith ops on tensors. Only non-elementwise ops (gather, scatter, concatenate, pad, slice, etc.) remain as stablehlo. |
| Large tile sizes produce large MLIR vectors (e.g., `vector<16x16xf32>`) | MLIR's vector lowering passes decompose large vectors into hardware-width operations. This is the same model `ShloToVectorPass` uses for dot products and reductions — no new risk. |
| Performance regression vs. current tiling + peeling approach | The current approach tiles elementwise ops to `kMaxVectorDim=8` and peels remainders. The new approach relies on MLIR's vector decomposition. Benchmark to verify. The tiled emitter already pads tiles to powers of 2, so remainder handling is rarely exercised today. |
| Ops between `StablehloLowerToArithPass` and `VectorizeElementwisePass` may not handle arith-on-tensor | Verified: `ShloToVectorPass` (matches stablehlo only), `Canonicalize` (generic), `LowerVectorMultiReduction` (matches vector only), `TensorOpsToBufferizablePass` (matches tensor.bitcast only), `StablehloLowerToXtilePass` (matches stablehlo only), `StablehloLegalizeToLinalgPass` (matches stablehlo only), `ConvertElementwise0DTensorToScalar` (matches rank-0 only). None conflict. |

## Files Affected

### Modified
- `xla/backends/cpu/codegen/fusion_compiler.cc` — pipeline reorder
  (remove 3 pass additions, add 2)
- `xla/backends/cpu/codegen/tiled/transforms/passes.td` — new pass def
- `xla/backends/cpu/codegen/tiled/transforms/passes.h` — new pass decl
- `xla/backends/cpu/codegen/tiled/transforms/BUILD` — new target

### New
- `xla/backends/cpu/codegen/tiled/transforms/vectorize_elementwise_pass.cc`
- `xla/backends/cpu/codegen/tiled/transforms/tests/vectorize_elementwise.mlir`

### Deleted (Phase 2, after verifying no other consumers)
- `xla/backends/cpu/codegen/tiled/transforms/linalg_elementwise_to_vector_pass.cc`
- `xla/backends/cpu/codegen/tiled/transforms/fuse_elementwise_pass.cc`
- `xla/backends/cpu/codegen/tiled/transforms/tests/linalg_elementwise_to_vector_pass.mlir`
- `xla/backends/cpu/codegen/tiled/transforms/tests/fuse_elementwise.mlir`
