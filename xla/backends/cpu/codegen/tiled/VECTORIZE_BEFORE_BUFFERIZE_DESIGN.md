# Design: Vectorize Before Bufferize in XLA:CPU Tiled Pipeline

## Summary

Three independent improvements to the XLA:CPU tiled emitter pipeline,
ordered from highest impact / lowest risk to lowest impact / highest
architectural change:

- **Phase 0 — Eliminate intermediate allocation + copy for the fusion
  result.** The single biggest source of wasted ops today. Likely a
  diagnosis of why `EmptyTensorEliminationPass` is failing to alias
  the final `tensor.empty` with the kernel's output buffer. Independent
  of when vectorization happens.

- **Phase 1 — Move `LinalgElementwiseToVectorPass` before
  bufferization.** Small, surgical reorder. Preserves linalg's tiling,
  peeling, and fusion machinery. Turns multi-use intermediates and
  chain intermediates into vector SSA values instead of memref allocs.

- **Phase 2 (conditional) — Replace the linalg path with a dedicated
  `VectorizeElementwisePass`.** Pursue only if Phase 1 data shows
  remaining wins from eliminating `linalg.generic` as an intermediate
  representation for elementwise ops.

The three phases are independent and can land in any order. Phase 0
should probably land first because it is almost certainly the source
of the observed ~50% op overhead on simple kernels.

## Motivation

### The observed overhead

For a simple `stablehlo.add` kernel on a single full tile, the
"right" op count is roughly: two loads, one add, one store to the
output buffer. Today we see ~50% more ops than that. The extra ops
come from two sources:

1. **An intermediate allocation + final copy.** The fusion result
   goes into a fresh `memref.alloca`, then a `memref.copy` (which
   `MemrefCopyToLoopsPass` lowers into a load+store loop) moves it
   into the kernel's output buffer. That is 1 extra load + 1 extra
   store per vector slot.

2. **For fusions with multiple elementwise ops, intermediate
   allocations between ops.** `FuseElementwisePass` handles the
   single-use case by merging `linalg.generic`s, but multi-use
   intermediates survive as separate allocations.

(1) is the bigger problem and affects every kernel including the
simplest ones. (2) only matters for multi-op fusions and only when
intermediates have multiple consumers.

### Why phase ordering matters

Vectorize-before-bufferize (Phases 1 and 2) addresses (2): chain
intermediates become vector SSA values, no allocation needed, and
multi-use is handled by SSA without a fusion pass.

Vectorize-before-bufferize does **not** address (1) by itself. The
final `vector.transfer_write` still targets a `tensor.empty`; whether
that `tensor.empty` gets collapsed into the output buffer depends on
empty-tensor-elimination / in-place bufferization / the emitter's
DPS structure — which is Phase 0.

### Convergence with the GPU pipeline

The GPU emitter operates on higher-level abstractions and lowers
tensors late. Moving CPU vectorization before bufferization follows
the same principle and opens the door to sharing more pass
infrastructure.

## Current Pipeline (Tiled Path)

From `fusion_compiler.cc : AddTiledOptimizationPasses` (lines 344–379):

```
RegisterOptimizationPasses
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
ConvertElementwiseToLinalgPass       # arith on tensor → linalg.generic
FuseElementwisePass                  # merge single-use linalg.generic chains
--- AddBufferizationPasses ---
LinalgElementwiseToVectorPass        # tile to kMaxVectorDim=8, peel, vectorize
FoldMemRefAliasOps
Canonicalize, CSE
```

`LinalgElementwiseToVectorPass` is doing real work: it tiles linalg
ops to `kMaxVectorDim=8`, generates `scf.for` loops, applies
`mlir::linalg::peelLoops()` for non-aligned remainders, then
vectorizes the inner tile bodies with `mlir::linalg::vectorize()`.
Any redesign must either preserve this functionality or have a
concrete replacement.

## Phase 0: Eliminate the intermediate alloc + copy for the fusion result

This is orthogonal to the vectorize-before-bufferize question and
should probably be tackled first. It likely captures the bulk of the
observed 50% op overhead.

### Hypothesis

For a simple `%r = stablehlo.add %a, %b` kernel, the current pipeline
generates IR like:

```mlir
// After vectorization + bufferization:
%alloca = memref.alloca() : memref<8x8xf32>
scf.for %i = 0 to 8 {
  %va = vector.transfer_read %a[%i, 0] : vector<8xf32>
  %vb = vector.transfer_read %b[%i, 0] : vector<8xf32>
  %vr = arith.addf %va, %vb
  vector.transfer_write %vr, %alloca[%i, 0]
}
memref.copy %alloca, %output_buf  // lowered to load+store loop
```

The desired IR is:

```mlir
scf.for %i = 0 to 8 {
  %va = vector.transfer_read %a[%i, 0] : vector<8xf32>
  %vb = vector.transfer_read %b[%i, 0] : vector<8xf32>
  %vr = arith.addf %va, %vb
  vector.transfer_write %vr, %output_buf[%i, 0]
}
```

The difference — one alloca, one copy loop — is exactly the ~50%
overhead.

### Why the alloca + copy survives

`AddBufferizationPasses` runs `EmptyTensorEliminationPass` before
`OneShotBufferize`. Its job is to replace `tensor.empty()` with a
destination from a DPS consumer, so that bufferization writes into
the destination buffer directly instead of allocating a new one.

If `EmptyTensorEliminationPass` doesn't fire for the fusion's final
result, the `tensor.empty` stays, bufferize allocates a buffer for
it, and a subsequent copy is required to move the result into the
actual output buffer.

Plausible reasons it doesn't fire:

1. **The output buffer isn't visible as a DPS destination** at the
   point the fusion result is produced. Whatever `LowerXTileEntryPass`
   generates needs to connect the output memref (or a tensor view of
   it) back to the DPS destination of the last op.

2. **`OneShotBufferize`'s in-place analysis is conservative** and
   refuses to alias due to a perceived potential for conflicting
   writes.

3. **The entry lowering explicitly emits `compute-into-temp +
   copy-to-output`.** In that case, no downstream pass will collapse
   it; the fix is in the emitter.

### Diagnosis plan

1. Build a minimal single-add kernel. Dump IR after every pass with
   `--mlir-print-ir-after-all`.
2. Find the first pass where an unrelated `memref.alloc`/`alloca`
   shows up, or where a `memref.copy` to the output appears. That
   identifies the culprit.
3. If the culprit is bufferization (the `tensor.empty` survives into
   bufferization and becomes an `alloc`), check whether the output
   destination is reachable from the `tensor.empty` in the IR. If
   not, adjust the entry lowering or insert
   `bufferization.materialize_in_destination` ops to make the link
   explicit.
4. If the culprit is the emitter itself, restructure it to write
   directly to the output.

### Fixes

Depending on what the diagnosis shows, one or more of:

- **Thread the output destination through the entry lowering.** Make
  sure the last op in the fusion has the output buffer (or a
  `to_tensor`-wrapped view of it) as its DPS destination.
- **Add an explicit `bufferization.materialize_in_destination`** at
  emit time for the fusion result. This is the canonical MLIR way to
  say "this tensor materializes into that buffer."
- **Pre-empt the problem: make `WriteVectorToTensor` destination-aware.**
  In Phase 2, when the transfer_write's target is known (e.g., the
  final result of a fusion), pass the output tensor in directly
  instead of creating a fresh `tensor.empty`.

### Success criteria

For a single-add kernel, final lowered IR should contain no
`memref.alloc`/`alloca` and no `memref.copy` to the output buffer.
Op count should approach the 4-op ideal per vector slot (two loads,
one op, one store).

## Phase 1: Move `LinalgElementwiseToVectorPass` before bufferization

### The change

One pass moves, nothing else changes:

```diff
   pm.addPass(CreateFuseElementwisePass());
+  pm.addPass(CreateLinalgElementwiseToVectorPass());
   AddBufferizationPasses(pm);
-  pm.addPass(CreateLinalgElementwiseToVectorPass());
   pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
```

### Why this works

Both `mlir::linalg::tileLinalgOp` and `mlir::linalg::vectorize` work
on tensor-based linalg ops as well as memref-based. On tensors they
produce `scf.for` with tensor iter_args, `tensor.extract_slice` /
`tensor.insert_slice` for the tile regions, and
`vector.transfer_read`/`vector.transfer_write` on tensors. All of
these have well-tested bufferization interfaces.

After bufferization, the `scf.for` with tensor iter_args becomes an
`scf.for` with memref iter_args (or better, in-place updates to a
single memref), `extract_slice`/`insert_slice` become
`memref.subview`, and the transfer ops operate on memrefs.

### What this captures

- **Chain intermediates.** Inside the tile loop body, chained
  elementwise ops become chains of vector SSA values. Intermediate
  tensors (from `tensor.insert_slice` of one op feeding
  `tensor.extract_slice` of the next) fold away through
  canonicalization, so bufferization sees no allocation to make.

- **Multi-use intermediates.** In the current pipeline these survive
  `FuseElementwisePass` and bufferize into separate memref allocs.
  After Phase 1 they are vector SSA values shared among all
  consumers.

### What this preserves

- **Tiling to `kMaxVectorDim=8`.** Same `mlir::linalg::tileLinalgOp`
  call, same tile sizes.
- **Peeling for non-aligned minor dims.** Same
  `mlir::linalg::peelLoops` call.
- **Vectorization of the inner tile.** Same `mlir::linalg::vectorize`
  call.
- **Cast-away-leading-ones canonicalization** and other vector
  cleanup.
- **Fusion.** `FuseElementwisePass` still runs before this pass, on
  tensor linalg ops, same as today.

### What this does *not* capture

- **The fusion-result alloca + copy** (Phase 0 concern).
- **The `linalg.generic` intermediate representation** itself, which
  Phase 2 addresses.

### Testing

1. Verify all existing XLA:CPU integration tests pass.
2. Check allocation counts for multi-op fusion patterns
   (especially multi-use intermediates). Should drop vs. baseline.
3. Check op counts for simple kernels. Should be unchanged or slightly
   lower; Phase 0 is what drives the bigger improvement here.

### Risks

| Risk | Mitigation |
|---|---|
| `mlir::linalg::tileLinalgOp` on tensors may generate suboptimal IR compared to memrefs | Both paths are exercised in upstream MLIR. If a concrete issue surfaces, file upstream. |
| Bufferization of vector-on-tensor + `scf.for` with tensor iter_args may have edge cases | `ShloToVectorPass` already produces vector-on-tensor IR today; bufferization handles it. `scf.for` with tensor iter_args is also well-supported. |
| Canonicalization between `LinalgElementwiseToVectorPass` and `AddBufferizationPasses` may be needed | Add a `createCanonicalizerPass` between them if the fold of `transfer_read ∘ transfer_write` doesn't happen automatically. |

## Phase 2 (conditional): Replace linalg for elementwise ops

Only pursue after Phase 1 ships and its impact is measured. If
Phase 1 closes the performance gap, this phase is optional
architectural cleanup. If measurable wins remain specifically from
the `linalg.generic` intermediate representation (e.g., large
multi-use intermediates not fully eliminated by SSA+canonicalization,
or code-size issues from the linalg → scf → vector lowering), then
proceed.

### The change

Replace three passes with one, plus move the vectorization earlier:

```diff
   pm.addPass(xtile::createConvertElementwise0DTensorToScalarPass());
-  pm.addPass(mlir::createConvertElementwiseToLinalgPass());
-  pm.addPass(CreateFuseElementwisePass());
+  pm.addPass(CreateVectorizeElementwisePass());
+  pm.addPass(mlir::createCanonicalizerPass());
   AddBufferizationPasses(pm);
-  pm.addPass(CreateLinalgElementwiseToVectorPass());
```

### `VectorizeElementwisePass`

A single rewrite pattern that matches any op with the `Elementwise`
trait on ranked tensor types (rank > 0):

```cpp
class VectorizeElementwisePattern : public mlir::RewritePattern {
  // Match: op with Elementwise trait, all operand/result types are
  //        ranked tensors with rank > 0.
  // Action:
  //   1. ReadTensorToVector each operand  (from lowering_utils.h)
  //   2. Clone the op with vector result types
  //   3. WriteVectorToTensor the result   (from lowering_utils.h)
  //      If the result's consumer has a known tensor destination
  //      (e.g., function return tied to an output buffer), target
  //      that destination instead of tensor.empty (Phase 0 hook).
};
```

### What this replaces

- **Tile + peel + vectorize on linalg** →  direct read-whole-tile-into-vector.
  The tiled emitter already pads tile dimensions to power-of-2 (see
  `tiled_fusion_emitter.cc:72-74` comment; verify this holds for all
  tilings), so peeling is unnecessary. `ConvertVectorToSCFPass` in
  the lowering phase decomposes multi-dim vector transfers into
  loops. LLVM legalization handles the SIMD decomposition of arith
  on large vectors.

- **`ConvertElementwiseToLinalgPass` + `FuseElementwisePass`** → implicit
  fusion through SSA value forwarding. Chained ops share vector SSA
  values directly; `transfer_read ∘ transfer_write` canonicalizes
  away intermediate writes.

### Prerequisites

Before committing to Phase 2, confirm:

1. **Tile minor dimensions are always power-of-2** in the tiled
   emitter. If not, Phase 2 needs a peeling or masking story, and
   the padding approach from earlier versions of this design may
   need to return.

2. **`ConvertVectorToSCFPass` + LLVM legalization produce code at
   least as good as the current tile+peel+vectorize path.** Measure
   with the cases we care about.

3. **Bufferization of vector-on-tensor IR scales** to the fusion
   patterns we produce (many chained ops, various shapes and dtypes).

### What we lose and how we compensate

| Lost capability | Compensation |
|---|---|
| MLIR-level peeling for non-aligned tile sizes | Rely on tile sizes being power-of-2. Verify and assert. |
| Explicit control of loop structure via `linalg.tileLinalgOp` | `ConvertVectorToSCFPass` decomposes multi-dim transfers. LLVM handles arith-on-large-vectors. |
| Upstream `linalg::populateElementwiseOpsFusionPatterns` fusion | SSA + `transfer_read ∘ transfer_write` canonicalization. |
| Cast-away-leading-ones canonicalization tied to `linalg::vectorize` | Add `populateCastAwayVectorLeadingOneDimPatterns` to the post-vectorize canonicalizer. |

### Risks

| Risk | Mitigation |
|---|---|
| LLVM's vector legalization produces worse SIMD code than MLIR's explicit tiling for our targets | Benchmark before committing. Phase 1 provides a fallback if Phase 2 regresses. |
| Non-power-of-2 tile sizes slip through and we get no peeling | Add an assertion in the tiled emitter that minor tile dims are power-of-2. Add a lit test. |
| Large MLIR vectors cause compile-time blowup in lowering or register pressure in generated code | `ShloToVectorPass` already exercises this pattern for dot/reduce/broadcast. Measure as part of prereq check. |

## Files Affected

### Phase 0
- `xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.{h,cc}` —
  possibly, if the entry lowering needs to thread the output
  destination.
- `xla/codegen/xtile/ir/transforms/lower_xtile_entry.cc` — possibly,
  same reason.
- Wherever the fusion-result-to-output wiring happens: probably
  inserting `bufferization.materialize_in_destination` or adjusting
  DPS output operands.

### Phase 1
- `xla/backends/cpu/codegen/fusion_compiler.cc` — one-pass reorder.

### Phase 2
- `xla/backends/cpu/codegen/fusion_compiler.cc` — pipeline edits.
- `xla/backends/cpu/codegen/tiled/transforms/passes.td`,
  `passes.h`, `BUILD` — register new pass.
- **New:**
  `xla/backends/cpu/codegen/tiled/transforms/vectorize_elementwise_pass.cc`
- **New test:**
  `xla/backends/cpu/codegen/tiled/transforms/tests/vectorize_elementwise.mlir`
- **Delete (after verifying no other consumers):**
  `linalg_elementwise_to_vector_pass.cc`,
  `fuse_elementwise_pass.cc`, and their tests.

## Recommended landing order

1. **Phase 0 first.** Diagnose the alloca + copy on a simple-add
   kernel. Fix. This should give the biggest op-count win
   immediately.
2. **Phase 1 next.** Low-risk reorder. Should close the gap for
   multi-op fusions with multi-use intermediates.
3. **Measure.** If Phases 0 + 1 hit the performance target, stop
   here.
4. **Phase 2 only if needed.** Larger architectural change, worth it
   only if the data justifies it.
