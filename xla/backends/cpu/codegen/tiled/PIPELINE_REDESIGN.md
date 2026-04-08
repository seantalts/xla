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

1. The xtile emitter (`xla/codegen/xtile/codegen/fusion_emitter.cc:1261`, via `EmitElementwise` at `xla/codegen/xtile/codegen/emitter_helpers.cc:364`) produces `stablehlo`/`arith`/`math` ops on *tile‑sized tensors* (e.g. `tensor<8x1024xf32>`).
2. `StablehloLowerToArithPass` rewrites the `stablehlo` elementwise ops to `arith`/`math` on those same tensors.
3. `ConvertElementwiseToLinalgPass` wraps each of those into a `linalg.generic` with a fresh `tensor.empty()` init operand.
4. `FuseElementwisePass` (`xla/backends/cpu/codegen/tiled/transforms/fuse_elementwise_pass.cc:50`) only fuses producers with exactly one use (`producer->hasOneUse()`). Anything with multiple uses, anything with a non‑elementwise op between the uses, and anything that reuses intermediate results stays un‑fused.
5. `OneShotBufferize` then allocates a separate memref for every un‑fused `tensor.empty()` / `linalg.generic` result — one `memref.alloc` per intermediate tile.
6. `LinalgElementwiseToVectorPass` finally rewrites each `linalg.generic` (now on memrefs) into an `scf.for` of `vector.transfer_read` → `arith.*` → `vector.transfer_write`. At this point the intermediate allocations are just a buffer ping‑pong that any downstream pass has to eliminate.

There is no allocation coming from the tiling emitter itself. `xtile::ExtractTileOp::bufferize` in `xla/codegen/xtile/ir/xtile_bufferization.cc:246` only allocates for the edge‑tile (padded) case — which is legitimate. All the "ton of allocations" are from `OneShotBufferize` on `linalg.generic` tile chains.

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
%out = vector.transfer_write %v3, %empty[0,0] : vector<8x1024xf32>, tensor<8x1024xf32>
xtile.insert_tile %out into %arg1 [...] : tensor<8x1024xf32>
```

`OneShotBufferize` now sees:
- `xtile.extract_tile` → bufferizes to a memref subview (with edge‑case padded alloc — unchanged).
- `vector.transfer_read` of the resulting tensor → folds through `bufferization::ToTensorOp` to read directly from the memref (MLIR handles this).
- The `arith`/`math`/`vector` chain in the middle → no tensors, nothing to bufferize, zero allocations.
- `vector.transfer_write` + `xtile.insert_tile` → bufferizes to a `transfer_write` directly into the output memref subview.

The only allocations left are the legitimate ones: padded edge tiles and whatever the multi‑reduction lowering needs.

## Migration strategy

Do it in steps, each landable on its own:

1. **Add the new `ElementwiseToVectorPass`** next to `shlo_to_vector.cc`. Standalone, tested with lit tests at `xla/backends/cpu/codegen/tiled/transforms/tests/elementwise_to_vector.mlir`. Don't wire it in yet.
2. **Add `LowerTranspose` to `ShloToVectorPass`'s pattern set** (`shlo_to_vector.cc:336`). Drive‑by; has its own test case.
3. **Move `StablehloLowerToXtilePass` and `StablehloLowerToArithPass` earlier**, before `ShloToVectorPass`. This is a reordering only — the IR invariant at the end of the pipeline doesn't change yet, but we front‑load all the shlo rewrites. Run the existing tests.
4. **Wire `ElementwiseToVectorPass` into the pipeline** after `ShloToVectorPass`, still *before* the linalg/bufferize/unvectorize sequence. At this point vectors flow through arith/math and the linalg path becomes a no‑op for anything the new pass handled. Run end‑to‑end. This is the "safe" checkpoint: if something slips through the new pass, the linalg fallback still handles it.
5. **Delete the linalg passes**: `StablehloLegalizeToLinalgPass`, `ConvertElementwiseToLinalgPass`, `FuseElementwisePass`, `LinalgElementwiseToVectorPass`. Also delete `FuseElementwisePass` and `LinalgElementwiseToVectorPass` source files and their tests, and drop the `mlir::linalg::registerBufferizableOpInterfaceExternalModels` registration and linalg‑related includes from `fusion_compiler.cc`. Measure allocation count reduction.
6. **Move `AddBufferizationPasses`** as late as possible (it's already last in the new layout). Confirm no intermediate allocations.
7. **Drop `TensorOpsToBufferizablePass`** if the `tensor.bitcast → arith.bitcast` rewrite it does is no longer needed once bitcasts arrive in vector form (it's likely already fine because `LowerBroadcastInDim` and friends don't produce `tensor.bitcast`, but verify).

After step 5 the pipeline no longer depends on `linalg` for tiled fusion. `linalg.h`/`Linalg.h` includes and the `LinalgDialect` registration in `CreateDialectRegistry` (`fusion_compiler.cc:611`) can be removed, reducing build time and the number of dialects the tiled context has to load.

## Things to verify before deleting anything

- **Elementwise ops the xtile emitter can produce that don't carry `OpTrait::Elementwise`**: audit `emitter_helpers.cc:364 EmitElementwise` and make sure every op it can create is handled. `stablehlo.compare`, `stablehlo.select`, `mhlo::reducePrecision<tensor::BitcastOp>` (line 438), `stablehlo.convert` — these need explicit patterns or explicit verification that the trait covers them.
- **Edge tiles**: `xtile.extract_tile` bufferization emits `scf.if` with a padded buffer allocation. Make sure `vector.transfer_read` of the `ToTensorOp` from either branch folds correctly. This is the main place where the new path could regress vs. the old path, because the old path had linalg working on memrefs and never had to cross the `tensor → ToTensorOp → vector.transfer_read` boundary.
- **Multi‑use tensors**: the current `FuseElementwisePass` bails on multi‑use. The new path handles multi‑use trivially because a shared vector SSA value just has multiple consumers — CSE and canonicalization keep it as a single value. Verify this with a fusion test that has a diamond.
- **Dynamic shapes / runtime‑sized tiles**: decide whether to handle or punt. Current tilings are always static (`IsVectorizable` in `linalg_elementwise_to_vector_pass.cc:63` rejects dynamic), so punting is fine initially.
- **Sub‑byte types**: `CreateUnpackSubByteVectorWritePass` runs in `AddTiledLoweringPasses`. `IsSupportedTilingType` in `tiled_fusion_emitter.cc:149` already excludes `<8`‑bit types, so this should be a no‑op for the tiled path. Double‑check.

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

## Success criteria

1. A lit test that exercises a chain of ≥ 3 elementwise ops on a single tile produces zero `memref.alloc` / `memref.alloca` in the final MLIR output (outside of the edge‑tile path).
2. `FusionCompiler::Compile` end‑to‑end tests for the tiled emitter still pass.
3. Generated LLVM IR for the benchmark fusions in `xla/backends/cpu/benchmarks` is the same or better (no extra memmoves, no extra allocas).
4. The tiled pass manager no longer registers `LinalgDialect` (checked via a compile‑time grep or a test).

## Integration with `cc_to_llvm_ir` and future tiled microkernel emitters

### The existing story

XLA has a mechanism to compile a C++ source file to LLVM IR bitcode at build time (`xla/codegen/intrinsic/cpp/cc_to_llvm_ir.bzl`, the `cc_ir_header` macro). The bitcode is embedded into the compiler as a string constant (e.g. `kTanhIr`, `kEigenUnaryIr`), and at JIT time it's spliced into the user's LLVM module so the implementations of the intrinsics can be inlined by LLVM itself. The C++ sources use clang `ext_vector_type` attributes (`Vec4f`, `Vec8f`, `Vec16f` in `xla/codegen/intrinsic/cpp/vector_ops.h:31`) that map 1‑to‑1 onto LLVM's `<N x float>` types, which in turn is what MLIR's `vector` dialect lowers to.

The pattern‑matching side lives in `xla/codegen/emitters/transforms/lower_xla_intrinsic_lib.cc`. `LowerXlaIntrinsicLibPass` walks `math.exp`, `math.erf`, `math.tanh`, `math.log1p`, `math.rsqrt`, and `arith.truncf` f32→bf16, and replaces each one with a `func.call` to the corresponding intrinsic function. It runs inside `AddGenericLoweringPasses` (`fusion_compiler.cc:237`), which is invoked at the end of both the scalar and tiled lowering pipelines.

### The constraint that matters for this redesign

**Every pattern in `LowerXlaIntrinsicLibPass` bails out on vectors of rank != 1** — see `lower_xla_intrinsic_lib.cc:94`, `:188`, `:221`. The patterns explicitly check:

```cpp
if (maybe_vector_type && maybe_vector_type.getRank() != 1) {
  return rewriter.notifyMatchFailure(op, "Vector rank is not 1.");
}
```

They even emit a warning — "Missed XLA intrinsic lowering as vector rank != 1" — because a multi‑dim vector `math.exp` will silently miss the embedded C++ lowering and fall through to whatever `createConvertMathToLLVMPass` or `createConvertMathToLibmPass` does instead (`fusion_compiler.cc:238`). That means today, if you write `math.exp : vector<8x1024xf32>` and hand it to `AddGenericLoweringPasses`, you'll get a libm call or an LLVM intrinsic with polynomial expansion — not the tuned Eigen‑based Vec8f path.

In the current pipeline this accidentally works out: `LinalgElementwiseToVectorPass` tiles along the minor dim to `kMaxVectorDim = 8` (`linalg_elementwise_to_vector_pass.cc:58`), wraps the tile in an `scf.for` nest, and the inner body has only 1‑D vectors of the register width. By the time `LowerXlaIntrinsicLibPass` sees the IR, every `math.*` op is already `vector<8xf32>` (or similar). The linalg detour was quietly doing the vector unrolling that the intrinsic library depends on.

The new `ElementwiseToVectorPass` proposed above does not do any tiling — it preserves the xtile tile shape (e.g. `tensor<8x1024xf32>` → `vector<8x1024xf32>`). If we wire it in without further changes, we'll lose the intrinsic library coverage for every multi‑dim fusion.

### The fix

Insert a vector unrolling pass between the new tiled optimization phase and the generic lowering phase, so that all elementwise `vector` ops are decomposed to 1‑D vectors of native register width before `LowerXlaIntrinsicLibPass` runs. MLIR already provides the machinery: `mlir::vector::populateVectorUnrollPatterns` with `UnrollVectorOptions().setNativeShape(...)` unrolls `arith.*`, `math.*`, `vector.transfer_read/write`, and friends to a given target shape.

Concrete placement in `AddTiledLoweringPasses` (`fusion_compiler.cc:384`):

```cpp
static void AddTiledLoweringPasses(mlir::OpPassManager& pm, bool fast_min_max) {
  pm.addPass(CreateVectorUnrollPass(/*native_vector_width=*/ ... ));   // NEW
  pm.addPass(CreateVectorToScalarPass());
  pm.addPass(cpu::CreateMemrefCopyToLoopsPass());
  pm.addPass(cpu::createLowerToLLVMPass());
  pm.addPass(mlir::createConvertVectorToSCFPass(...));
  ...
}
```

The new pass can be a thin wrapper around `populateVectorUnrollPatterns` — roughly 30 lines, mirroring the existing `VectorToScalarPass` file layout. The native shape comes from `FusionCompiler::Options::vector_width`, which is already plumbed in (`fusion_compiler.cc:299`).

Note that `createConvertVectorToSCFPass` does *not* do this work for us. It unrolls multi‑dim `vector.transfer_read`/`transfer_write` ops into loops that read 1‑D slices, but it leaves the elementwise `arith.addf : vector<8x1024xf32>` alone (it stitches the 1‑D reads back into a multi‑dim SSA value via `vector.insert_strided_slice`). That's why the new unroll pass has to precede it.

### Microkernel emitters in the new pipeline

With `cc_to_llvm_ir` the pattern for a new tiled microkernel — say a hand‑tuned 16×16 f32 matmul, or a vectorized transpose, or a fused gelu — is:

1. Write the C++ implementation against `Vec<N><T>` types in `xla/codegen/intrinsic/cpp/<name>.cc`.
2. Add a `cc_ir_header` BUILD rule that compiles it to embeddable LLVM IR.
3. Add a C++ class under `xla/codegen/intrinsic/` that knows the function signature and holds the embedded IR.
4. Add a new `mlir::OpRewritePattern` in `LowerXlaIntrinsicLibPass` (or a sibling pass) that matches the specific vector op shape and replaces it with `func.call @<intrinsic_name>`.

The vectorize‑first pipeline makes step 4 substantially simpler. Today the same work would need to pattern‑match:

| Phase | What the IR looks like for a 16×16 matmul tile |
|---|---|
| After `ShloToVectorPass` | `vector.contract` with inputs produced by `vector.transfer_read` on tensors |
| After `StablehloLegalizeToLinalg` + `ConvertElementwiseToLinalg` | `linalg.matmul` / `linalg.generic` with tensor operands |
| After `OneShotBufferize` | `linalg.matmul` on memrefs + surrounding allocs |
| After `LinalgElementwiseToVector` | `scf.for` nest with 1‑D `vector.transfer_read` + `vector.contract` of a smaller shape |

Depending on when you run your microkernel pattern, the input IR has radically different structure. In the new pipeline it's always the same: a small number of `vector.*` and `arith.*`/`math.*` ops on tile‑shaped vectors, surrounded by `xtile.extract_tile`/`xtile.insert_tile` at the boundary. Pattern‑matching `vector.contract : vector<16x16xf32>` is trivial and doesn't care about bufferization state.

There are two places a microkernel lowering could plug in:

1. **Before unrolling (tile‑shape level)** — match `vector.contract` of the full tile shape (e.g. 16×16). This is where a tiled matmul microkernel belongs: the C++ implementation consumes entire tiles and the pattern fires before any sub‑tile unrolling. Requires the microkernel function signature to accept wide vectors (e.g. `Vec256f` for 16×16 f32, or a pair of vector pointers).
2. **After unrolling (register‑width level)** — match `math.exp : vector<8xf32>` etc. This is where the *existing* intrinsic library plugs in. Unchanged by this redesign, except that it now runs after our new unrolling pass instead of after the linalg pipeline.

If (1) fires, its output is either a `func.call` returning a wide vector (which the unrolling pass then leaves alone — unrolling elementwise ops doesn't unroll opaque calls) or a `func.call` returning a memref which reintroduces bufferization state. The first option is cleaner and fits naturally into the vector SSA flow.

### What to change (and what not to)

- **No changes to `cc_to_llvm_ir.bzl`**, the `cc_ir_header` macro, the embedding machinery, or any existing `.cc` intrinsic source. They're orthogonal.
- **No changes to the existing 1‑D intrinsic patterns** in `lower_xla_intrinsic_lib.cc`. They continue to run unchanged in `AddGenericLoweringPasses`.
- **Add** a `VectorUnrollPass` at the top of `AddTiledLoweringPasses`, parameterized by the native vector width from `FusionCompiler::Options`.
- **Optionally add** (future work, not blocking this redesign): a tile‑shape microkernel lowering pass that runs after `ElementwiseToVectorPass`/`ShloToVectorPass` in `AddTiledOptimizationPasses`. This is where future 16×16 matmul or transpose microkernels written with `cc_to_llvm_ir` would be pattern‑matched. For the first landing of this redesign, leave this out — it's not needed to achieve the allocation‑elimination goal.

### Summary

The integration story is: **the redesign doesn't break the existing `cc_to_llvm_ir` intrinsic library as long as we add a vector unrolling pass between the tiled optimization phase and the generic lowering phase.** Beyond that, the vectorize‑first IR is a strictly better substrate for future tiled microkernel emitters because every potential insertion point sees the same flat vector/SSA structure instead of different dialect mixtures depending on pipeline stage.
