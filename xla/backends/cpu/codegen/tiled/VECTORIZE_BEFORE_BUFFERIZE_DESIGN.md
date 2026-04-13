# Design: Vectorize Before Bufferize in XLA:CPU Tiled Pipeline

## Summary

Reorder the XLA:CPU tiled emitter pipeline so that vectorization of
elementwise operations happens **before** bufferization, not after. This
eliminates the round-trip through arith for elementwise ops, reduces
unnecessary intermediate allocations, brings the CPU pipeline closer to
the GPU pipeline's structure, and removes several passes that exist only
to bridge the current ordering.

## Motivation

### 1. Unnecessary allocations for elementwise ops

The tiling emitter currently produces one `tensor.empty()` per
intermediate elementwise result. `FuseElementwisePass` merges chains
with single-use intermediates into a single `linalg.generic`, but
multi-use intermediates survive. After bufferization each surviving
`tensor.empty()` becomes a `memref.alloc()`. Vectorizing *before*
bufferization means the intermediate values live in vector registers
(SSA values) instead of memory, so there is nothing to allocate.

### 2. Round-trip through arith and back up

The current pipeline lowers StableHLO elementwise ops to arith (via
`StablehloLowerToArithPass`), then lowers everything to linalg, then
bufferizes, and *then* vectorizes linalg back into vector ops that
contain the same arith ops. The net effect is:

```
stablehlo.add (tensor)
  → arith.addf (tensor)          # StablehloLowerToArithPass
    → linalg.generic { arith.addf }  # stablehlo-legalize-to-linalg
      → bufferize (memref)
        → vector.transfer_read + arith.addf + vector.transfer_write
                                        # LinalgElementwiseToVectorPass
```

By vectorizing first we go directly:

```
stablehlo.add (tensor<...>)
  → vector ops on tensor            # new: vectorize at stablehlo/tensor level
    → bufferize (memref)
```

### 3. Convergence with the GPU pipeline

The GPU emitter never goes through linalg for elementwise work. It
operates on higher-level abstractions and lowers tensors late. Moving
CPU vectorization before bufferization follows the same principle and
opens the door to sharing more of the pass infrastructure.

## Current Pipeline (Tiled Path)

From `fusion_compiler.cc : AddTiledOptimizationPasses`:

```
RegisterOptimizationPasses          # shared: simplify-arith, canon, CSE
LowerXTileEntryPass
StablehloTargetIndependentOptimization
StablehloLowerToArithPass           # shlo elementwise → arith on tensors  [REMOVE]
ShloToVectorPass                    # shlo dot/reduce/broadcast/iota → vector
Canonicalize
LowerVectorMultiReduction
TensorOpsToBufferizablePass
StablehloLowerToXtilePass
StablehloLegalizeToLinalgPass       # remaining shlo → linalg.generic      [CHANGE]
ConvertElementwise0DTensorToScalar
ConvertElementwiseToLinalgPass      # arith-on-tensor → linalg.generic     [REMOVE]
FuseElementwisePass                 # merge linalg.generics                [REMOVE]
--- bufferization ---
LinalgElementwiseToVectorPass       # linalg.generic → vector r/w + arith  [REMOVE]
FoldMemRefAliasOps
Canonicalize, CSE
```

## Proposed Pipeline

```
RegisterOptimizationPasses          # shared: simplify-arith, canon, CSE
LowerXTileEntryPass
StablehloTargetIndependentOptimization
ShloToVectorPass                    # dot/reduce/broadcast/iota → vector  [KEEP, extend]
VectorizeElementwisePass            # shlo elementwise → vector on tensor [NEW]
Canonicalize
LowerVectorMultiReduction
TensorOpsToBufferizablePass
StablehloLowerToXtilePass
StablehloLegalizeToLinalgPass       # non-elementwise shlo → linalg       [KEEP, reduced scope]
--- bufferization ---
FoldMemRefAliasOps
Canonicalize, CSE
```

### Passes removed

| Pass | Why it can go |
|---|---|
| `StablehloLowerToArithPass` | Elementwise ops are vectorized directly from stablehlo; no need to go through arith on tensors first. |
| `ConvertElementwiseToLinalgPass` | No more arith-on-tensor ops to convert. |
| `FuseElementwisePass` | Fusion is no longer needed: vectorized elementwise chains are already SSA vector values with no intermediate tensor materializations. |
| `LinalgElementwiseToVectorPass` | Vectorization now happens before bufferization; the post-bufferization linalg vectorizer is unnecessary. |

### Passes added / changed

| Pass | Description |
|---|---|
| **`VectorizeElementwisePass`** (new) | Lowers stablehlo elementwise ops (`add`, `mul`, `sub`, `div`, `max`, `min`, `and`, `or`, `xor`, `rem`, `round_nearest_even`, etc.) to vector dialect ops on tensors. For each elementwise op: (1) `vector.transfer_read` each operand from the input tensor, (2) apply the corresponding arith/math op on vectors, (3) `vector.transfer_write` the result back to an output tensor. Tile dimensions are chosen based on the target's vector register width. |
| **`ShloToVectorPass`** (extended) | Already handles `dot_general`, `reduce`, `broadcast_in_dim`, `iota`. Could optionally absorb the elementwise lowering or remain separate for modularity. |
| `StablehloLegalizeToLinalgPass` | Scope narrows: only needs to handle non-elementwise, non-vectorized ops (e.g. `gather`, `scatter`, `concatenate`, `pad`, `slice`). |

## Handling Partial Tiles with Padding

The current pipeline handles partial tiles by **loop peeling**: the
vectorizer tiles to `kMaxVectorDim` (8), generates a main loop for full
tiles, then peels a remainder loop with a smaller power-of-2 vector.

The proposed pipeline uses **padding** instead:

1. **Pad the input tensors** to the next multiple of the chosen vector
   width along the vectorized dimension (the minor dimension). Use
   `tensor.pad` with an appropriate identity element (0 for add, 1 for
   mul, -inf for max, etc.).
2. **Vectorize the padded tensor** — every iteration is now a full-width
   vector operation. No remainder loop, no peeling, no masking.
3. **Extract the original slice** of the result with `tensor.extract_slice`
   (or let bufferization + downstream passes handle the truncation if
   the output buffer already has the correct size).

Benefits over peeling:
- Simpler generated IR (one loop body instead of main + remainder).
- The padded region is dead after extraction, so the optimizer can
  often remove it entirely when tile sizes are statically known.
- Follows the same approach used by GPU/Triton (`GetPaddedTileSizes`
  pads to power-of-2) — shared infrastructure opportunity.

When the tensor dimension is statically known and already a multiple of
the vector width, no padding is inserted. When the dimension is dynamic,
the pad amount is computed as `(vector_width - dim % vector_width) %
vector_width`.

## Detailed Walkthrough: Elementwise Add

### Before (current pipeline)

```
// Input: stablehlo
%r = stablehlo.add %a, %b : tensor<8x100xf32>

// After StablehloLowerToArithPass:
%r = arith.addf %a, %b : tensor<8x100xf32>

// After ConvertElementwiseToLinalgPass:
%r = linalg.generic {
  ^bb0(%lhs: f32, %rhs: f32, %out: f32):
    %s = arith.addf %lhs, %rhs : f32
    linalg.yield %s
} ins(%a, %b) outs(%empty)

// After bufferization:
// %alloc = memref.alloc() : memref<8x100xf32>   ← allocation
// linalg.generic ... ins(%buf_a, %buf_b) outs(%alloc)

// After LinalgElementwiseToVectorPass (tiles to 8, peels remainder of 4):
// Main loop (0..96 step 8):
//   %va = vector.transfer_read %buf_a[%i, %j] : vector<8xf32>
//   %vb = vector.transfer_read %buf_b[%i, %j] : vector<8xf32>
//   %vr = arith.addf %va, %vb : vector<8xf32>
//   vector.transfer_write %vr, %alloc[%i, %j]
// Peeled loop (96..100 step 4):
//   %va = vector.transfer_read ... : vector<4xf32>
//   ...
```

### After (proposed pipeline)

```
// Input: stablehlo
%r = stablehlo.add %a, %b : tensor<8x100xf32>

// After VectorizeElementwisePass:
// Pad to 104 (next multiple of 8):
%a_padded = tensor.pad %a low[0,0] high[0,4] { ... 0.0 ... }
                 : tensor<8x100xf32> to tensor<8x104xf32>
%b_padded = tensor.pad %b low[0,0] high[0,4] { ... 0.0 ... }
                 : tensor<8x100xf32> to tensor<8x104xf32>

// Vectorized loop over padded dimension (0..104 step 8), fully uniform:
scf.for %j = 0 to 104 step 8 {
  scf.for %i = 0 to 8 step 1 {
    %va = vector.transfer_read %a_padded[%i, %j] : vector<8xf32>
    %vb = vector.transfer_read %b_padded[%i, %j] : vector<8xf32>
    %vr = arith.addf %va, %vb : vector<8xf32>
    vector.transfer_write %vr, %out_padded[%i, %j]
  }
}
%r = tensor.extract_slice %out_padded[0,0][8,100][1,1]
       : tensor<8x104xf32> to tensor<8x100xf32>

// After bufferization:
// The padded tensors may share buffers with the originals if the
// downstream consumer only reads [0:100]. Bufferization's alias
// analysis + empty-tensor elimination can often elide the padding
// allocation entirely when the slice is immediately extracted.
//
// In the worst case a small stack-promoted alloc covers the padding.
// No per-element intermediate allocation for the elementwise op itself.
```

When the minor dimension (100) is statically known, the pad amount (4)
is a compile-time constant and the `tensor.extract_slice` can be folded
by canonicalization. When the dimension is already a multiple of 8, no
padding ops are generated at all.

## Implementation Plan

### Phase 1: VectorizeElementwisePass

1. Create `xla/backends/cpu/codegen/tiled/transforms/vectorize_elementwise_pass.cc`.
2. For each supported stablehlo elementwise op, implement a rewrite
   pattern that:
   - Reads operand tensors into vectors via `vector.transfer_read`
     (tiled to `kMaxVectorDim` on the minor dimension).
   - Pads the input tensors to the next multiple of the vector width
     when the minor dimension is not evenly divisible (using
     `tensor.pad` with op-specific identity elements).
   - Applies the corresponding arith/math op on vectors.
   - Writes the result back via `vector.transfer_write`.
   - Extracts the original (unpadded) slice if padding was applied.
3. Handle type promotion for bf16/f16 div/rem (currently done in
   `StablehloLowerToArithPass`; move to the new pass).
4. Handle unsigned integer → signless conversion (same).
5. Register the pass in `passes.td` and `passes.h`.

### Phase 2: Pipeline reorder

1. In `AddTiledOptimizationPasses`:
   - Remove `StablehloLowerToArithPass`.
   - Add `VectorizeElementwisePass` after `ShloToVectorPass`.
   - Remove `ConvertElementwiseToLinalgPass`.
   - Remove `FuseElementwisePass`.
   - Remove `LinalgElementwiseToVectorPass` (post-bufferization).
2. `StablehloLegalizeToLinalgPass` remains for non-elementwise ops.
3. Verify that `AddBufferizationPasses` handles vector-on-tensor IR
   correctly (one-shot bufferize should work — vector transfer ops on
   tensors are bufferizable by default in MLIR).

### Phase 3: Cleanup & testing

1. Add MLIR filecheck tests for `VectorizeElementwisePass`:
   - Statically-shaped tensors, evenly divisible.
   - Statically-shaped tensors, not evenly divisible (padding case).
   - Dynamic shapes.
   - All supported element types (f32, f16, bf16, i32, u32, etc.).
   - Type promotion for bf16/f16 division.
2. Run existing XLA:CPU integration tests end-to-end.
3. Benchmark allocation counts for representative elementwise fusion
   patterns (the motivating case).
4. Delete dead code: `linalg_elementwise_to_vector_pass.cc`,
   `fuse_elementwise_pass.cc` and their BUILD targets, if no other
   pipeline uses them.

### Phase 4: Extend to more ops (optional)

- Absorb `ShloToVectorPass` lowerings (dot, reduce, broadcast, iota)
  into a unified `VectorizeStablehloPass` if it simplifies the pipeline
  further.
- Share padding infrastructure with GPU's `GetPaddedTileSizes`.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| One-shot bufferize may not handle vector-on-tensor IR well | MLIR's bufferization framework supports `vector.transfer_read/write` on tensors natively. Verify with small examples first. |
| Padding may introduce unnecessary copies | Canonicalization + empty-tensor elimination should fold extract_slice(pad(x)) → x when padding is zero. For non-zero padding, the overhead is small and bounded by `vector_width - 1` elements per dimension. |
| `StablehloLegalizeToLinalgPass` may try to lower ops we already vectorized | Run `VectorizeElementwisePass` first; remaining stablehlo ops (non-elementwise) are legal input to the linalg pass. Alternatively, add a filter/control function to skip already-lowered ops. |
| Performance regression from padding vs. peeling | Padding avoids branch misprediction in the remainder loop and uses uniform vector widths. Benchmark to confirm. For very small tensors the padding overhead is proportionally larger — but these are already handled by the scalar path, not the tiled path. |
| Multi-use intermediates that currently survive as linalg.generics | In the new pipeline these become vector SSA values. If a value is used by multiple consumers, MLIR's SSA semantics handle it naturally — no allocation needed. This is strictly better. |

## Files Affected

### Modified
- `xla/backends/cpu/codegen/fusion_compiler.cc` — pipeline reorder
- `xla/backends/cpu/codegen/tiled/transforms/passes.td` — new pass definition
- `xla/backends/cpu/codegen/tiled/transforms/passes.h` — new pass declaration

### New
- `xla/backends/cpu/codegen/tiled/transforms/vectorize_elementwise_pass.cc`
- `xla/backends/cpu/codegen/tiled/transforms/tests/vectorize_elementwise.mlir`

### Deleted (after migration)
- `xla/backends/cpu/codegen/tiled/transforms/linalg_elementwise_to_vector_pass.cc`
- `xla/backends/cpu/codegen/tiled/transforms/fuse_elementwise_pass.cc`
- `xla/backends/cpu/codegen/tiled/transforms/tests/linalg_elementwise_to_vector_pass.mlir`
- `xla/backends/cpu/codegen/tiled/transforms/tests/fuse_elementwise.mlir`
- `xla/codegen/xtile/ir/transforms/lower_stablehlo_to_arith.cc` (if no other consumer)
