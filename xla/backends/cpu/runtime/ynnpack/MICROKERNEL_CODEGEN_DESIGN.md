# Design: LLVM-IR microkernel codegen for the XLA:CPU ynnpack path

Status: draft / RFC
Scope: XLA:CPU backend, `__ynn_fusion` path

## 1. Goal

Today XLA:CPU offloads matched fusions (dot / conv / reduce / elementwise) to
**ynnpack** — the next-gen graph API inside the XNNPACK fork — and lets ynnpack
*execute* them at runtime. This proposal keeps ynnpack's op coverage and
kernel-selection knowledge but changes the back half: instead of handing the
optimized subgraph to ynnpack's runtime to execute, we **emit LLVM IR
microkernels** for the fusion, JIT-compile them through XLA's existing
`JitCompiler`, and run them with a plain `KernelThunk`.

The win is not "LLVM is faster than XNNPACK's microkernels." It's everything
*around* the microkernel that we currently can't touch:

- **Epilogue/prologue fusion**: fold the surrounding elementwise ops
  (bias, activation, scale, cast) into the GEMM/conv kernel and apply them to
  each output tile *in registers*, before it is stored. Today the elementwise
  ops live in the ynn subgraph but execute as separate slinky pipeline stages
  over materialized buffers.
- **Shape/target specialization**: loop bounds become compile-time constants,
  K-loops unroll, the exact `-mcpu`/feature set XLA already knows about
  (`TargetMachineFeatures`) is baked in. No generic runtime dispatch.
- **One runtime, one threadpool**: parallelism comes from XLA's workgroup grid
  (`KernelThunk` → Eigen intra-op pool), not from a second scheduler
  (slinky) bridged through `SlinkyThreadPool`. The slinky interpreter and the
  threadpool shim leave the hot path.
- **AOT + object cache**: the kernel is ordinary XLA object code, so it
  participates in AOT compilation and the executable object cache. The current
  path re-builds and re-optimizes a ynn subgraph + runtime per executable
  instantiation.
- **Autotuning**: tile parameters can be exposed to the existing
  `LlvmKernelAutotuner`.

## 2. How the path works today

```
HLO module
  │  LibraryRewriter (HLO pass)  +  YnnMatcher
  │    - matches kDot/kConvolution/kReduce/kReduceWindow + elementwise neighbors
  │    - wraps them in HloFusionInstruction (kCustom),
  │      FusionBackendConfig.kind = "__ynn_fusion"  (kYnnFusionKind)
  ▼
HLO fusion (__ynn_fusion)
  │  ThunkEmitter::EmitYnnFusionThunk()
  │    - collects arg/result BufferAllocation::Slices
  │    - marks constant operands as "captured" (for weight packing)
  │    - EmitYnnFusionBuilder(computation, captured_ids)  → subgraph builder
  ▼
YnnFusionThunk  (Thunk::Kind::kYnnFusion)
  │  Execute():
  │    - EmitYnnSubgraph(): walk fusion in post-order, ynn_define_{tensor,dot,
  │      unary,binary,reduce,convolution,...}, then ynn_optimize_subgraph()
  │    - ynn_create_runtime(subgraph, threadpool)  → ynn_reshape_runtime()
  │    - ynn_set_external_value_data() to bind buffers
  │    - ynn_invoke_runtime()      ← slinky interprets the schedule and calls
  │                                   precompiled XNNPACK microkernels,
  │                                   parallelizing via SlinkyThreadPool→Eigen
  ▼
results
```

Key fact that shapes this whole design: **neither ynnpack nor slinky emits
code.** XNNPACK ships precompiled (asm / SIMD-intrinsic) microkernels; slinky is
a *runtime interpreter* for data-flow pipelines ("Slinky is not responsible for
generating the inner loop code... all operations that read/write buffers are
user-defined callbacks"). ynnpack's "optimize" step is **kernel selection,
graph fusion, and weight packing** — not codegen. So an LLVM-IR path cannot just
"ask ynnpack for IR"; the IR has to be produced by a new emitter. The open
design question is *who owns the microkernel IR* (Section 5).

Relevant seams (file:symbol):

- `xla/backends/cpu/transforms/library_rewriter.cc` : `LibraryRewriter`,
  `CreateLibraryFusion`, `FuseNeighbors`
- `xla/backends/cpu/transforms/ynn_matcher.h` : `YnnMatcher` (`SupportedOps`,
  `IsOpSupported`, `ShouldCreateFusion`, `fusion_kind`)
- `xla/backends/cpu/ynn_support.{h,cc}` : `IsDotSupportedByYnn`,
  `IsElementwiseOpSupportedByYnn`, `IsConvolutionOpSupportedByYnn`, ...
- `xla/backends/cpu/ynn_emitter.cc` : `EmitYnnSubgraph`, `Define*Op`
- `xla/service/cpu/thunk_emitter.cc` : `EmitYnnFusionThunk` (dispatch at the
  `kYnnFusionKind` check)
- `xla/backends/cpu/runtime/ynnpack/ynn_fusion_thunk.cc` : runtime execute path
- flag: `xla_cpu_experimental_ynn_fusion_type` (repeated `LibraryFusionType`),
  consumed in `cpu_compiler.cc`

## 3. Target architecture

Reuse the front half (detection + fusion). Replace the back half (subgraph
builder + runtime thunk) with an LLVM-IR emitter that feeds XLA's existing
JIT + kernel runtime.

```
HLO fusion (__ynn_fusion)            ← unchanged front half
  │  ThunkEmitter: when microkernel-codegen is enabled, route to ...
  ▼
YnnKernelEmitter : KernelEmitter<LlvmKernelSource>     ← NEW
  │  1. Plan with ynnpack: build the ynn subgraph, ynn_optimize_subgraph(),
  │     query the chosen schedule + per-op microkernel/tile parameters.
  │  2. EmitKernelPrototype() — KernelApiIrBuilder gives the XLA_CPU kernel
  │     ABI function (call frame, arg/result IrArrays, workgroup id/dims).
  │  3. Emit the loop nest + microkernel bodies + fused epilogue into the
  │     llvm::Module (Section 5 = where the microkernel body comes from).
  │  4. Return KernelDefinition{ KernelSpec(name, NumWorkGroups, buffers),
  │                              LlvmKernelSource(module) }.
  ▼
JitCompiler (+ IrCompiler: opt level, target features, fast-math)
  │  → ObjectLoader → CompiledFunctionLibrary (symbol → XLA_CPU_Kernel*)
  ▼
KernelThunk (Thunk::Kind::kKernel)   ← reuses XLA's existing kernel runtime
  │  Execute(): resolve symbol once, launch NumWorkGroups across the
  │             Eigen intra-op pool (Kernel::Launch).
  ▼
results
```

Everything from `JitCompiler` down already exists and is exactly what the
non-library dot/elementwise emitters use (`DotKernelEmitter`,
`ElementalKernelEmitter`). The new surface area is the `YnnKernelEmitter` and
the microkernel body source.

## 4. Why this maps cleanly onto existing XLA infra

- **Kernel packaging is already abstracted.** `KernelEmitter<LlvmKernelSource>`
  → `KernelDefinition{KernelSpec, LlvmKernelSource}` is the same contract
  `DotKernelEmitter` and `ElementalKernelEmitter` satisfy. Nothing downstream
  needs to know the kernel came from a ynn fusion.
- **The kernel ABI is fixed and simple.** `KernelApiIrBuilder::EmitKernelPrototype`
  emits the `XLA_CPU_KernelCallFrame` function: flat `{void* data, size_t size}`
  args (arguments then results), plus `workgroup_id` / `num_workgroups`. The
  emitter writes a function body against `KernelPrototype.arguments` /
  `.results` (as `llvm_ir::IrArray`) and partitions work by `workgroup_id`.
- **Parallelism is the workgroup grid.** This is the natural replacement for
  slinky's parallel-for: pick a partition of the output (M-tiles × N-tiles)
  into `NumWorkGroups{x,y}` and emit a kernel that computes its tile from
  `workgroup_id`. `DotKernelEmitter` already does exactly this (it returns a
  `DotOpWorkGroupDim` and indexes with `workgroup_id.x/y`). `KernelThunk` then
  launches across the Eigen pool — the same pool `SlinkyThreadPool` wraps today,
  minus the shim.
- **Microkernel scaffolding exists.** `tiled_dot_emitter.cc`
  (`EmitSmallGemm`, `EmitRowMajorGemv`, `EmitColumnMajorGemv`, `MemoryTile`) and
  `VectorIrBuilder` (FMA, vector load/store/broadcast, horizontal reductions,
  tile variables) are already an LLVM-IR microkernel toolkit. There is also an
  MLIR tiling path (`tiled/tiled_fusion_emitter.h: EmitTiledFusionKernel`,
  the `xla_cpu` dialect, `fusion_compiler`) that is a viable alternative
  emission target (Section 5, option A2).

## 5. The core decision: where does the microkernel IR come from?

ynnpack/slinky don't emit IR, so we must. Three options, increasing fidelity to
XNNPACK's hand-tuned kernels and increasing cost.

### Option A — XLA emits the microkernels; ynnpack is the planner
ynnpack runs only as an oracle: build + `ynn_optimize_subgraph`, then read back
the decisions (microkernel variant, register-tile `MR×NR`, vector width, loop
order, packing layout, K-unroll). XLA emits the GEMM/conv inner loop in LLVM IR
for those parameters.

- **A1 (LLVM IR directly):** extend `tiled_dot_emitter` / `VectorIrBuilder`
  with register-blocked, packed GEMM/conv microkernels parameterized by the
  ynn-chosen tile. Most code, but no new toolchain and fully target-aware via
  `TargetMachineFeatures`.
- **A2 (MLIR tiling dialect):** emit the ynn-planned schedule into the existing
  `xla_cpu`/tiling MLIR and lower via `fusion_compiler`. Reuses XLA's tiling +
  vectorization passes; less hand-written IR; couples to the MLIR emitter's
  maturity.

Pros: clean dependency story; native fusion + specialization; reuses XLA infra.
Cons: must re-derive XNNPACK's microkernel quality in LLVM/MLIR (perf risk and
effort), especially for quantized and conv paths.

### Option B — ynnpack microkernels as LLVM bitcode templates  (recommended seed)
Compile XNNPACK's C/SIMD-intrinsic microkernels to LLVM **bitcode** at build
time (they are already generated from templates under XNNPACK `tools/`). At JIT
time: ynnpack selects the microkernel; XLA emits a thin driver loop nest +
fused epilogue, **links/inlines the selected bitcode microkernel**,
constant-propagates the dims, and runs the normal IR pipeline. Inlining +
specialization + epilogue fusion happen in LLVM.

Pros: preserves XNNPACK kernel quality; still gets fusion, specialization, AOT.
Cons: needs a bitcode build of XNNPACK microkernels and an LLVM-version
coupling; asm-only microkernels are excluded (fall back for those).

### Option C — new LLVM backend inside ynnpack/slinky
`ynn_emit_llvm(subgraph, target) → llvm::Module`. Conceptually cleanest
("ynnpack generates the IR") but the largest change to a third-party dep, heavy
LLVM coupling upstream, and the hardest to land/maintain. Not recommended as a
starting point.

**Recommendation:** start with **A1** for `f32`/`bf16` GEMM + epilogue (smallest
blast radius, reuses `tiled_dot_emitter`/`VectorIrBuilder`, proves the
end-to-end path), and adopt **B** for kernels where emitted IR underperforms
XNNPACK (quantized GEMM, conv). Keep ynnpack as the planner in both. This lets
us ship a vertical slice early and improve fidelity kernel-by-kernel.

## 6. Supporting design points

### 6.1 Epilogue / prologue fusion (the main payoff)
The `__ynn_fusion` already groups dot/conv with its elementwise neighbors into
one HLO computation. The emitter walks that computation: emit the
contraction microkernel producing an output tile in registers, then emit the
elementwise chain (`Define*Op` equivalents, via `CpuElementalIrEmitter`
generators like `ElementalKernelEmitter` uses) **on the tile**, then store.
This removes the intermediate materialization between contraction and
elementwise that the slinky pipeline pays for today.

### 6.2 Weight packing / constants
Today constant weights are "captured" so ynnpack can pack them once
(`CapturingBuilder` + `captured_arguments_ids`). In the codegen path, for
*static constant* weights prefer **compile-time prepack**: run ynnpack's packing
routine at compile time and materialize a new packed constant buffer that the
kernel reads directly (no per-call repack). Options for the mechanics: a
dedicated one-shot pack `KernelThunk`, or fold the packed bytes into a constant
allocation. Non-constant operands either repack in-kernel (a prologue stage) or
gate the op back to the runtime path.

### 6.3 Selection, gating, and fallback
- Reuse `YnnMatcher` / `ynn_support` predicates unchanged for *what* fuses.
- Add a debug flag, e.g. `xla_cpu_experimental_ynn_codegen` (or extend
  `LibraryFusionType` with codegen variants), to choose runtime vs codegen per
  fusion type. Default off; flip on per kernel class as it matures.
- The emitter returns `absl::Status`; on any unsupported construct
  (dynamic shapes, asm-only microkernel, type/layout the emitter can't handle)
  **fall back to `YnnFusionThunk`** for that fusion. Both thunks coexist; this
  is an incremental, reversible rollout.

### 6.4 Dynamic shapes
The single-fused-static-kernel model assumes compile-time shapes (the common
XLA:CPU case). For dynamic shapes, either emit a kernel with runtime loop bounds
(less unrolling/specialization) or fall back to the runtime path initially.

### 6.5 Threading
Drop `SlinkyThreadPool`/`ynn_threadpool` from the hot path; parallelism is the
`KernelThunk` workgroup launch over the Eigen intra-op pool. The emitter must
choose a workgroup partition that balances tiles across cores (mirrors the
parallel dim slinky picks today). Keep the slinky shim only for fusions that
still fall back to `YnnFusionThunk`.

## 7. Phasing

1. **Skeleton + fallback.** `YnnKernelEmitter` that handles the simplest matched
   dot (e.g. `f32` GEMM, no epilogue) via A1, behind a default-off flag; route
   in `ThunkEmitter`; everything else falls back to `YnnFusionThunk`. Land the
   end-to-end JIT→KernelThunk path and numerics tests.
2. **Epilogue fusion.** Fold elementwise neighbors onto the output tile; this is
   where the perf delta over the runtime path should appear. Add bf16.
3. **Packing.** Compile-time prepack of constant weights.
4. **Coverage / fidelity.** Adopt Option B (bitcode microkernels) for quantized
   GEMM and conv; broaden type/layout support; optionally A2 (MLIR) for tiling.
5. **Autotuning.** Expose tile params to `LlvmKernelAutotuner`.

## 8. Risks / open questions

- **Microkernel quality (Option A).** Matching XNNPACK's tuned kernels in
  emitted IR is non-trivial; benchmark early to decide A vs B per kernel class.
- **Bitcode toolchain (Option B).** Building/serving XNNPACK microkernel
  bitcode and pinning LLVM versions across XLA and the bitcode build.
- **Packing correctness/perf** for non-constant operands.
- **Coverage parity.** Conv, reduce-window, quantized, and the full elementwise
  set are a long tail; the fallback keeps this safe but parity takes time.
- **Maintenance vs the runtime path.** We carry both until codegen reaches
  parity; the flag + fallback make that explicit and reversible.

## 9. Decisions needed before implementation

1. Microkernel IR source to seed with: **A1** (recommended), A2, or B.
2. Runtime model: single fused static kernel (recommended) vs also supporting a
   JIT'd-inner-callback mode that keeps slinky as orchestrator.
3. Scope of this task: design only, or land the Phase-1 vertical slice.
