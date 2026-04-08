# XLA:CPU Tiled Pipeline Redesign

## Goal

Simplify the XLA:CPU tiled fusion pipeline by:
1. Vectorizing **before** bufferization (currently the other way around).
2. Skipping the detour through `linalg` entirely (`stablehlo → linalg → bufferize → vector`).
3. Eliminating the "down to `arith`, back up to `linalg`, back down to `vector`" zig-zag.
4. Bringing the CPU tiled pipeline closer to the GPU loop emitter pipeline.

Concrete target: remove the allocations the tiled emitter currently emits for elementwise ops that don't need them.

## Current Pipeline

The tiled pipeline is built in `xla/backends/cpu/codegen/fusion_compiler.cc:344` (`AddTiledOptimizationPasses`). It runs in this order:

| # | Pass | Dialect transition |
|---|---|---|
| 1 | `RegisterOptimizationPasses` | common opts |
| 2 | `CreateLowerXTileEntryPass` | xtile entry |
| 3 | `createStablehloTargetIndependentOptimizationPass` | shlo opts |
| 4 | `xtile::createStablehloLowerToArithPass` | **shlo → arith/math** (on tensors) |
| 5 | `CreateShloToVectorPass` | **shlo.{dot_general,reduce,broadcast,iota} → vector** |
| 6 | canonicalize |  |
| 7 | `createLowerVectorMultiReductionPass` | vector |
| 8 | `CreateTensorOpsToBufferizablePass` | tensor |
| 9 | `xtile::createStablehloLowerToXtilePass` | shlo → xtile specializations |
| 10 | **`createStablehloLegalizeToLinalgPass`** | **remaining shlo → linalg** |
| 11 | `xtile::createConvertElementwise0DTensorToScalarPass` | unwrap 0‑D tensors |
| 12 | **`createConvertElementwiseToLinalgPass`** | **arith/math on tensors → linalg.generic** |
| 13 | **`CreateFuseElementwisePass`** | fuse linalg.generic chains |
| 14 | **`AddBufferizationPasses`** | **tensor → memref (this is where the allocs come from)** |
| 15 | **`CreateLinalgElementwiseToVectorPass`** | **linalg on memrefs → scf.for + vector transfers** |
| 16 | canonicalize / CSE |  |

The problem lives in **10 → 12 → 13 → 14 → 15**. The pipeline takes elementwise ops that are already `arith`/`math` on tile tensors (produced by pass 4), pushes them back up into `linalg.generic`, tries to fuse them, bufferizes the whole thing (creating one allocation per un‑fused tensor result), and only *then* unwinds it back to vectors.

Pass 4 (`StablehloLowerToArithPass`, `xla/codegen/xtile/ir/transforms/lower_stablehlo_to_arith.cc:139`) has already done the interesting work: after it runs, `stablehlo.add tensor<8x1024xf32>` is already `arith.addf tensor<8x1024xf32>`. The lift back to `linalg.generic` in pass 12 is pure pipeline overhead.

## Why the allocations happen

Allocations come from two interacting sources. Fixing the pipeline order only removes half of them directly; the other half stops firing as a side‑effect.

### Source 1: `OneShotBufferize` on `linalg.generic` chains

1. The xtile emitter (`xla/codegen/xtile/codegen/fusion_emitter.cc:1261`, via `EmitElementwise` at `xla/codegen/xtile/codegen/emitter_helpers.cc:364`) produces `stablehlo`/`arith`/`math` ops on *tile‑sized tensors* (e.g. `tensor<8x1024xf32>`).
2. `StablehloLowerToArithPass` rewrites the `stablehlo` elementwise ops to `arith`/`math` on those same tensors.
3. `ConvertElementwiseToLinalgPass` wraps each of those into a `linalg.generic` with a fresh `tensor.empty()` init operand.
4. `FuseElementwisePass` (`xla/backends/cpu/codegen/tiled/transforms/fuse_elementwise_pass.cc:50`) only fuses producers with exactly one use (`producer->hasOneUse()`). Anything with multiple uses, anything with a non‑elementwise op between the uses, and anything that reuses intermediate results stays un‑fused.
5. `OneShotBufferize` then allocates a separate memref for every un‑fused `tensor.empty()` / `linalg.generic` result — one `memref.alloc` per intermediate tile.
6. `LinalgElementwiseToVectorPass` finally rewrites each `linalg.generic` (now on memrefs) into an `scf.for` of `vector.transfer_read` → `arith.*` → `vector.transfer_write`. At this point the intermediate allocations are just a buffer ping‑pong that any downstream pass has to eliminate.

### Source 2: the `xtile` bufferization interface itself

`ExtractTileOp` in `xla/codegen/xtile/ir/xtile_bufferization.cc` is conservative in several ways that all push `OneShotBufferize` toward inserting allocations:

- `bufferizesToAllocation` returns `true` (`:221`) with the comment "we must be conservative." The analysis treats every `extract_tile` result as if it were a freshly‑allocated buffer.
- `isWritable` returns `false` (`:241`). The tile tensor cannot be written through in place.
- `bufferize` itself calls `memref::AllocOp::create` in two places:
  - `:271`, the full‑tile path: whenever the subview has a non‑identity layout, it allocates a default‑layout memref and copies into it (there's a `TODO(willfroom)` to remove this).
  - `:196`, `GetPaddedTileBuffer`, the edge‑tile path: allocates a padded buffer and copies the clamped subview in. This one is legitimate and unavoidable.
- `InsertTileOp::bufferizesToAllocation` is also `true` (`:307`), with the same conservative comment.

The `isWritable = false` + `bufferizesToAllocation = true` combination is what makes the current pipeline especially bad. `linalg.generic` with a `tensor.empty()` init operand is precisely the shape of op that writes *through* its output tensor — and the init in practice aliases or needs to be proven disjoint from the extract_tile result that feeds the chain. The analysis can't prove in‑place reuse against a not‑writable conservative allocation, so it inserts a fresh `memref.alloc` plus a copy at each boundary. Combined with source 1, every un‑fused `linalg.generic` in a tile chain ends up with its own `memref.alloc`, regardless of whether the chain could logically reuse one buffer.

### Why the redesign still helps

In vector‑SSA form, none of the chain in the middle is a tensor. The only tensors `OneShotBufferize` sees at the boundary are `xtile.extract_tile → vector.transfer_read` (which folds to a memref‑level transfer read of the subview) and `vector.transfer_write → xtile.insert_tile` (which folds to a memref transfer write into the destination subview). Nothing in the chain is writing *through* the extract_tile tensor, so `isWritable = false` and `bufferizesToAllocation = true` stop causing problems in practice even though the interface stays conservative.

The non‑identity‑layout alloc at `:271` still fires when it has to, but it's rare and out of scope for this redesign — it's a separate `TODO(willfroom)` already noted in the source. The padded edge‑tile alloc at `:196` stays and is correct.

## GPU loop pipeline (reference)

`AddLoopTransformationPasses` in `xla/backends/gpu/codegen/emitters/emitter_base.cc:494` does **not** use `StablehloLegalizeToLinalg`, `ConvertElementwiseToLinalg`, `OneShotBufferize`, or any linalg/bufferization pipeline (confirmed by grep — zero matches under `xla/backends/gpu`). It goes directly from xla/emitters ops to `scf`, then `emitters::CreateLowerTensorsPass` lowers tensors to raw pointer loads/stores. GPU never leaves tensor land until the final pointer‑level lowering.

For the CPU tiled path we can't go *that* far — `xtile::ExtractTileOp`/`InsertTileOp` need memref subviews to drive contiguous loads. But everything *between* the tile reads and tile writes can stay on SSA vectors and never touch `linalg` or `bufferize`.

## Proposed Pipeline

```
AddTiledOptimizationPasses:
  RegisterOptimizationPasses
  LowerXTileEntryPass
  StablehloTargetIndependentOptimizationPass
  StablehloLowerToXtilePass                    # specialized xtile lowerings
  StablehloLowerToArithPass                    # shlo elementwise → arith/math on tensors
  ConvertElementwise0DTensorToScalarPass       # drop tensor<f32> → f32 for elementwise
  ShloToVectorPass                             # shlo {dot,reduce,broadcast,iota,transpose} → vector
  --- NEW: ElementwiseToVectorPass ---         # arith/math on tile tensors → vector ops
  canonicalize / CSE
  LowerVectorMultiReductionPass
  TensorOpsToBufferizablePass
  AddBufferizationPasses                       # now only bufferizes xtile boundary ops
  canonicalize / CSE
```

Deleted from the pipeline:
- `createStablehloLegalizeToLinalgPass`
- `createConvertElementwiseToLinalgPass`
- `CreateFuseElementwisePass`
- `CreateLinalgElementwiseToVectorPass`

The invariant after the new `ElementwiseToVectorPass` is: **every intermediate tile value is a `vector<...>` SSA value**. The only remaining tensor values are at the `xtile.extract_tile` / `xtile.insert_tile` boundaries, where bufferization still needs to turn them into memref subviews.

## The new pass: `ElementwiseToVectorPass`

The core conversion is small because most of the machinery already exists.

**Type converter**: `tensor<SxT>` → `vector<SxT>` (static shapes only).

**Source materialization** (vector → tensor when a legacy consumer still wants a tensor): `vector.transfer_write` into a fresh `tensor.empty()`. Same helper as `WriteVectorToTensor` in `xla/backends/cpu/codegen/tiled/transforms/lowering_utils.cc:66`.

**Target materialization** (tensor → vector when a legacy producer still emits a tensor): `vector.transfer_read` from the tensor at zero indices. Same helper as `ReadTensorToVector` in `lowering_utils.cc:44`.

**Pattern**: a single `OpTraitConversionPattern<OpTrait::Elementwise>` that clones the op with converted operand/result types. `arith.addf`, `math.exp`, `stablehlo.select`, etc. all carry the `Elementwise` trait and are polymorphic over tensor/vector, so cloning with new types is sufficient.

This is the exact same structure as `ConvertElementwise0DTensorToScalarPass` at `xla/codegen/xtile/ir/transforms/convert_elementwise_0d_tensor_to_scalar_pass.cc:42`, just with a different type converter. That file is the template for the new pass.

Notes on tricky cases:
- **`arith.constant`**: needs its own `OpConversionPattern` to rewrite `dense<...> : tensor<SxT>` into `dense<...> : vector<SxT>` (mirrors the constant pattern in the 0‑D pass).
- **Dynamic shapes**: mark the op illegal only when all shapes are static, leave dynamic tensors untouched (they will fall back through the current path, or fail, depending on how conservative we want to be initially).
- **Converts / reshapes / broadcasts / transposes on elementwise chains**: `arith.extf`/`arith.truncf`/`arith.bitcast`/`arith.sitofp` etc. are all elementwise and are handled for free. `tensor.from_elements` (0‑D) and `tensor.extract` need either a materialization or a small helper pattern.
- **`ShloToVectorPass` already handles the non‑elementwise ops** (`dot_general`, `reduce`, `broadcast_in_dim`, `iota`), and it already uses `ReadTensorToVector`/`WriteVectorToTensor` at its boundaries. Its output interleaves fine with the new pass: at boundaries it reads a tensor to a vector (doing work), at the next elementwise op the vector is consumed directly (no more `transfer_read` needed after canonicalization).
- **`LowerTranspose` in `shlo_to_vector.cc:209` is defined but never added to the pattern set** (see line 336 — only `LowerDotGeneral`, `LowerReduce`, `LowerBroadcastInDim`, `LowerIota` are added). Worth adding as a drive‑by fix so transposes also stay in vector land.

## What about bufferization?

After the new pass, the IR at the boundary of `AddBufferizationPasses` looks like:

```mlir
%tile = xtile.extract_tile %arg0 [...] : tensor<8x1024xf32>
%v0 = vector.transfer_read %tile[0,0], %pad : tensor<8x1024xf32>, vector<8x1024xf32>
%v1 = arith.addf %v0, %v0 : vector<8x1024xf32>
%v2 = math.exp %v1 : vector<8x1024xf32>
%v3 = arith.mulf %v2, %v0 : vector<8x1024xf32>
%empty = tensor.empty() : tensor<8x1024xf32>
%out = vector.transfer_write %v3, %empty[0,0] : vector<8x1024xf32>, tensor<8x1024xf32>
xtile.insert_tile %out into %arg1 [...] : tensor<8x1024xf32>
```

The middle of the chain — `%v0 → %v1 → %v2 → %v3` — is pure SSA vectors with no tensors. `OneShotBufferize` has literally nothing to bufferize for it, because none of its operands or results are tensors. That part is unambiguous.

The `vector.transfer_read %tile` on the extract_tile result folds through `bufferization.to_tensor` to a direct memref‑level transfer read of the subview during bufferization. `vector.transfer_read` implements `BufferizableOpInterface` in upstream MLIR and doesn't need a tensor copy.

### The N→1 claim (definite)

For an elementwise chain of length N that the current pipeline bufferizes to N `memref.alloc`s (one per un‑fused `linalg.generic` with its own `tensor.empty()` init), the new pipeline produces exactly one remaining `tensor.empty()` — the one `WriteVectorToTensor` (`lowering_utils.cc:66`) creates at the very end of the chain, to hold the final vector before it hits `xtile.insert_tile`. Everything in between is vectors. This is true regardless of fusion shape: diamonds, multi‑use values, and non‑elementwise ops in the middle of the chain stop breaking things, because the SSA vector form has none of the tensor aliasing issues that `FuseElementwisePass` bails on.

**N allocs → 1 alloc, mechanically, for any chain shape.** That's the main win.

### The remaining 1 alloc (eliminated by giving `InsertTileOp` `SubsetInsertionOpInterface`)

Whether that last `tensor.empty()` itself bufferizes to an allocation depends on whether `createEmptyTensorEliminationPass` (`fusion_compiler.cc:314`) can trace backward from a `SubsetInsertionOpInterface` op through the SSA chain and find the empty. The algorithm lives at `mlir/lib/Dialect/Bufferization/Transforms/EmptyTensorElimination.cpp:128‑188`: walk every `SubsetInsertionOpInterface` in the op, take `getSourceOperand()`, then `findValueInReverseUseDefChain` backward with `followEquivalentOnly = true` until a `tensor::EmptyOp` is found, then call `buildSubsetExtraction()` on the insertion op to produce a view of the destination and rewrite the empty's use.

For our chain `%empty → vector.transfer_write → xtile.insert_tile → %dest`, the relevant facts (verified against the MLIR source):

- **`vector.transfer_write` is a `DestinationStyleOpInterface` op** (`mlir/include/mlir/Dialect/Vector/IR/VectorOps.td:1500`) and its bufferization uses `DstBufferizableOpInterfaceExternalModel` (`mlir/lib/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.cpp:74`). That base class returns `{result, BufferRelation::Equivalent}` from `getAliasingValues` when queried about the DPS init operand (`mlir/include/mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h:39‑46`), so walking *backward* from the result via `defaultGetAliasingOpOperands` (`mlir/lib/Dialect/Bufferization/IR/BufferizableOpInterface.cpp:946`) produces `{dest_operand, Equivalent}`. The reverse use‑def walk with `followEquivalentOnly = true` therefore steps cleanly from the transfer_write result to its destination tensor operand.
- **`vector.transfer_write` is also registered as a `SubsetInsertionOpInterface`** via `vector::registerSubsetOpInterfaceExternalModels` (`mlir/lib/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.cpp:44‑66`, registered at `fusion_compiler.cc:630`), but its `buildSubsetExtraction` is a `TODO: Implement when needed` that returns `Value()`. Using `vector.transfer_write` directly as the elimination anchor therefore does nothing (`EmptyTensorElimination.cpp:168` skips on null replacement). That's fine for our case — we don't need it to be the anchor, only the pass-through step.
- **`xtile.insert_tile` does not implement `SubsetInsertionOpInterface`.** It only implements `TiledBufferInterface` (`xtile_ops.td:36,196`), which extends `BufferizableOpInterface`. Empty tensor elimination never visits it, so the chain is not anchored anywhere for our final empty.

With the dialect as it stands, `OneShotBufferize` allocates a fresh memref for the surviving `tensor.empty`, `vector.transfer_write` writes into that memref, and `InsertTileOp::bufferize` (`xtile_bufferization.cc:344`) emits a `bufferization::MaterializeInDestinationOp` that lowers (at `mlir/lib/Dialect/Bufferization/IR/BufferizationOps.cpp:604‑607`) to a `memref.copy` from the fresh alloc into the destination subview. `PromoteBuffersToStackPass` (`fusion_compiler.cc:334`, 4 KB cap) turns the alloc into an `alloca` for most tile sizes and `BufferHoisting` (`:319`) hoists it out of enclosing loops, but it's still a scratch buffer and a tile‑sized copy per fusion.

### Getting to zero: teach `InsertTileOp` `SubsetInsertionOpInterface`

Add `SubsetInsertionOpInterface` to `xtile_ops.td:196` and implement its methods in `xtile_bufferization.cc`. The implementation is a direct copy of `InsertSliceLikeOpSubsetInsertionOpInterface` in `mlir/lib/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.cpp:47‑83`:

- `getSourceOperand` → `getSourceMutable()`
- `getDestinationOperand` → `getDestinationMutable()`
- `buildSubsetExtraction(builder, loc)` → `tensor::ExtractSliceOp::create(builder, loc, sourceType, getDestination(), getMixedOffsets(), getMixedSizes(), getMixedStrides())`
- `getValuesNeededToBuildSubsetExtraction` → the dynamic offsets, sizes, strides, plus the destination value

With that interface in place, empty tensor elimination runs its walk, visits `xtile.insert_tile`, gets `%out` as the source operand, `findValueInReverseUseDefChain` follows the `Equivalent` edge from `%out` through `vector.transfer_write` to `%empty`, finds the `tensor::EmptyOp`, and rewrites the empty's use at the transfer_write's destination operand to a `tensor.extract_slice` of `%dest` at the insert_tile's offsets. After canonicalization the IR becomes:

```mlir
%slice = tensor.extract_slice %dest [off] [sz] [1] : ...
%out   = vector.transfer_write %v, %slice[0,0] : ...
xtile.insert_tile %out into %dest [off] [sz] [1]
```

`OneShotBufferize` now bufferizes `%slice` to a memref subview of `%dest` (no alloc), `vector.transfer_write` writes in place into the subview (no alloc), and the chain has zero scratch allocations in the middle.

**One remaining subtlety**: `InsertTileOp::bufferize` still emits a `MaterializeInDestinationOp` whose source and target now both resolve to the same subview of `%dest`. `MaterializeInDestinationOp::bufferize` (`BufferizationOps.cpp:588‑612`) unconditionally calls `options.createMemCpy(rewriter, loc, srcBuffer, buffer)` — there's no `srcBuffer == buffer` short‑circuit. The resulting `memref.copy` with identical source and destination is a semantic no‑op and LLVM should fold it, but there's no MLIR‑level canonicalization that guarantees removal. Cleanest fix: have `InsertTileOp::bufferize` detect the in‑place case (the source's bufferized value is already the destination subview) and skip emitting the materialize entirely, which is a small conditional added to the existing `scf.if` branches at `xtile_bufferization.cc:339‑359`. Worth doing as part of step 8 but not strictly blocking — an unneeded `memref.copy` is not the same as a `memref.alloc`.

**N→1→0, composed from two orthogonal changes: the pipeline redesign gets us from N to 1, and the `SubsetInsertionOpInterface` implementation gets us from 1 to 0.** The two changes can land independently; the redesign is the high‑value part and doesn't depend on the interface work. The interface work is useful on its own — any other future lowering that writes into `insert_tile` also benefits.

The only allocations that remain after both changes are the legitimate ones: padded edge tiles in the `else` branch of `ExtractTileOp::bufferize` (`xtile_bufferization.cc:196`), non‑identity‑layout copies in the `then` branch (`:271`, already a `TODO(willfroom)`), and whatever the multi‑reduction lowering emits via `vectorized_reduce_emitter.cc:222` directly.

## Migration strategy

Do it in steps, each landable on its own:

1. **Add the new `ElementwiseToVectorPass`** next to `shlo_to_vector.cc`. Standalone, tested with lit tests at `xla/backends/cpu/codegen/tiled/transforms/tests/elementwise_to_vector.mlir`. Don't wire it in yet.
2. **Add `LowerTranspose` to `ShloToVectorPass`'s pattern set** (`shlo_to_vector.cc:336`). Drive‑by; has its own test case.
3. **Move `StablehloLowerToXtilePass` and `StablehloLowerToArithPass` earlier**, before `ShloToVectorPass`. This is a reordering only — the IR invariant at the end of the pipeline doesn't change yet, but we front‑load all the shlo rewrites. Run the existing tests.
4. **Wire `ElementwiseToVectorPass` into the pipeline** after `ShloToVectorPass`, still *before* the linalg/bufferize/unvectorize sequence. At this point vectors flow through arith/math and the linalg path becomes a no‑op for anything the new pass handled. Run end‑to‑end. This is the "safe" checkpoint: if something slips through the new pass, the linalg fallback still handles it.
5. **Delete the linalg passes**: `StablehloLegalizeToLinalgPass`, `ConvertElementwiseToLinalgPass`, `FuseElementwisePass`, `LinalgElementwiseToVectorPass`. Also delete `FuseElementwisePass` and `LinalgElementwiseToVectorPass` source files and their tests, and drop the `mlir::linalg::registerBufferizableOpInterfaceExternalModels` registration and linalg‑related includes from `fusion_compiler.cc`. Measure allocation count reduction.
6. **Move `AddBufferizationPasses`** as late as possible (it's already last in the new layout). Confirm no intermediate allocations.
7. **Drop `TensorOpsToBufferizablePass`** if the `tensor.bitcast → arith.bitcast` rewrite it does is no longer needed once bitcasts arrive in vector form (it's likely already fine because `LowerBroadcastInDim` and friends don't produce `tensor.bitcast`, but verify).
8. **Teach `InsertTileOp` `SubsetInsertionOpInterface`** so that the final `tensor.empty()` at the end of each vectorized chain gets eliminated by `createEmptyTensorEliminationPass`. This is the 1→0 step and is entirely orthogonal to the pipeline redesign — it can land before, during, or after the other steps. Implementation goes in `xla/codegen/xtile/ir/xtile_ops.td:196` (add the trait) and `xla/codegen/xtile/ir/xtile_bufferization.cc` (the four interface methods: `getSourceOperand`, `getDestinationOperand`, `buildSubsetExtraction`, `isEquivalentSubset`).

After step 5 the pipeline no longer depends on `linalg` for tiled fusion. `linalg.h`/`Linalg.h` includes and the `LinalgDialect` registration in `CreateDialectRegistry` (`fusion_compiler.cc:611`) can be removed, reducing build time and the number of dialects the tiled context has to load.

## Things to verify before deleting anything

- **Elementwise ops the xtile emitter can produce that don't carry `OpTrait::Elementwise`**: audit `emitter_helpers.cc:364 EmitElementwise` and make sure every op it can create is handled. `stablehlo.compare`, `stablehlo.select`, `mhlo::reducePrecision<tensor::BitcastOp>` (line 438), `stablehlo.convert` — these need explicit patterns or explicit verification that the trait covers them.
- **Edge tiles**: `xtile.extract_tile` bufferization emits `scf.if` with a padded buffer allocation. Make sure `vector.transfer_read` of the `ToTensorOp` from either branch folds correctly. This is the main place where the new path could regress vs. the old path, because the old path had linalg working on memrefs and never had to cross the `tensor → ToTensorOp → vector.transfer_read` boundary.
- **Multi‑use tensors**: the current `FuseElementwisePass` bails on multi‑use. The new path handles multi‑use trivially because a shared vector SSA value just has multiple consumers — CSE and canonicalization keep it as a single value. Verify this with a fusion test that has a diamond.
- **Dynamic shapes / runtime‑sized tiles**: decide whether to handle or punt. Current tilings are always static (`IsVectorizable` in `linalg_elementwise_to_vector_pass.cc:63` rejects dynamic), so punting is fine initially.
- **Sub‑byte types**: `CreateUnpackSubByteVectorWritePass` runs in `AddTiledLoweringPasses`. `IsSupportedTilingType` in `tiled_fusion_emitter.cc:149` already excludes `<8`‑bit types, so this should be a no‑op for the tiled path. Double‑check.
- **1‑D vector requirement of `LowerXlaIntrinsicLibPass`**: every pattern in `xla/codegen/emitters/transforms/lower_xla_intrinsic_lib.cc` bails on vectors of rank != 1 (`:94`, `:188`, `:221`) and emits a "Missed XLA intrinsic lowering as vector rank != 1" warning. Today this works out because `LinalgElementwiseToVectorPass` tiles the minor dim to `kMaxVectorDim = 8` and leaves only 1‑D vectors inside the generated `scf.for`. `ElementwiseToVectorPass` as proposed preserves the tile shape (e.g. `vector<8x1024xf32>`), so we need a `vector` unrolling pass (thin wrapper around `mlir::vector::populateVectorUnrollPatterns` with native shape from `FusionCompiler::Options::vector_width`) inserted at the top of `AddTiledLoweringPasses`, before `AddGenericLoweringPasses` runs the intrinsic library lowering. `createConvertVectorToSCFPass` doesn't do this — it unrolls transfer ops but leaves `arith.addf : vector<8x1024xf32>` alone.

## Files touched

| Action | File |
|---|---|
| **Add** | `xla/backends/cpu/codegen/tiled/transforms/elementwise_to_vector_pass.cc` |
| **Add** | `xla/backends/cpu/codegen/tiled/transforms/tests/elementwise_to_vector.mlir` |
| **Edit** | `xla/backends/cpu/codegen/tiled/transforms/passes.h` / `passes.td` — declare new pass |
| **Edit** | `xla/backends/cpu/codegen/tiled/transforms/BUILD` — register new source |
| **Edit** | `xla/backends/cpu/codegen/tiled/transforms/shlo_to_vector.cc:336` — add `LowerTranspose` |
| **Edit** | `xla/backends/cpu/codegen/fusion_compiler.cc:344` (`AddTiledOptimizationPasses`) — new pipeline |
| **Edit** | `xla/backends/cpu/codegen/fusion_compiler.cc` — drop linalg includes and dialect registration |
| **Delete** | `xla/backends/cpu/codegen/tiled/transforms/fuse_elementwise_pass.cc` + test |
| **Delete** | `xla/backends/cpu/codegen/tiled/transforms/linalg_elementwise_to_vector_pass.cc` + test |
| **Edit** | `xla/codegen/xtile/ir/xtile_ops.td:196` — add `SubsetInsertionOpInterface` to `InsertTileOp` (step 8) |
| **Edit** | `xla/codegen/xtile/ir/xtile_bufferization.cc` — implement the four `SubsetInsertionOpInterface` methods (step 8) |

## Success criteria

1. A lit test that exercises a chain of ≥ 3 elementwise ops on a single tile produces at most one `memref.alloc` / `memref.alloca` in the final MLIR output after step 5, and zero after step 8 (outside of the edge‑tile path).
2. `FusionCompiler::Compile` end‑to‑end tests for the tiled emitter still pass.
3. Generated LLVM IR for the benchmark fusions in `xla/backends/cpu/benchmarks` is the same or better (no extra memmoves, no extra allocas).
4. The tiled pass manager no longer registers `LinalgDialect` (checked via a compile‑time grep or a test).

