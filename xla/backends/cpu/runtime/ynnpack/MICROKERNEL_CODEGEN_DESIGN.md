# Design: LLVM-IR microkernel codegen for the XLA:CPU ynnpack path

Status: draft / RFC (v3)
Scope: XLA:CPU backend, `__ynn_fusion` path

## 1. Goal

Today XLA:CPU offloads matched fusions (dot / conv / reduce / elementwise) to
**ynnpack** — the next-gen graph API inside the XNNPACK fork — and lets ynnpack
*execute* them at runtime. This proposal changes the back half: **get
ynnpack's microkernels out as LLVM IR**, embed that IR into a fused kernel that
XLA JIT-compiles through its existing `JitCompiler`, and run it with a plain
`KernelThunk` — instead of handing the subgraph to ynnpack's runtime to execute.

The win is not "LLVM beats XNNPACK's microkernels" — we *keep* XNNPACK's
microkernels, now as IR. The win is everything the offload model hides from the
compiler once the microkernel is IR in our module:

- **Epilogue/prologue fusion** — fold surrounding elementwise ops (bias,
  activation, scale, cast) onto each output tile *in registers*, instead of
  separate pipeline stages over materialized buffers.
- **Shape/target specialization** — the ukernel inlines into a driver with
  compile-time loop bounds; K unrolls; the exact `-mcpu`/feature set XLA tracks
  (`TargetMachineFeatures`) is applied.
- **One runtime, one threadpool** — parallelism is XLA's workgroup grid
  (`KernelThunk` → Eigen pool), not slinky bridged through `SlinkyThreadPool`.
- **AOT + object cache** — the kernel is ordinary XLA object code; the current
  path rebuilds a ynn subgraph + runtime per executable.
- **Autotuning** — driver tile params can feed `LlvmKernelAutotuner`.

## 2. How the path works today

```
HLO module
  │  LibraryRewriter (HLO pass) + YnnMatcher
  │    matches kDot/kConvolution/kReduce/kReduceWindow + elementwise neighbors,
  │    wraps them in HloFusionInstruction (kCustom),
  │    FusionBackendConfig.kind = "__ynn_fusion"  (kYnnFusionKind)
  ▼
HLO fusion (__ynn_fusion)
  │  ThunkEmitter::EmitYnnFusionThunk()
  │    collects arg/result slices, marks constant operands "captured",
  │    EmitYnnFusionBuilder(computation, captured_ids) → subgraph builder
  ▼
YnnFusionThunk (Thunk::Kind::kYnnFusion)
  │  Execute():
  │    EmitYnnSubgraph(): post-order walk → ynn_define_{tensor,dot,unary,...},
  │                       ynn_optimize_subgraph()
  │    ynn_create_runtime() → ynn_reshape_runtime()
  │    ynn_set_external_value_data() (bind buffers)
  │    ynn_invoke_runtime()  ← slinky interprets the schedule, calls precompiled
  │                            XNNPACK microkernels, parallel via SlinkyThreadPool
  ▼
results
```

Two facts shape this design:

1. **Neither ynnpack nor slinky emits code.** XNNPACK ships precompiled
   (asm / SIMD-intrinsic) microkernels; slinky is a runtime *interpreter*
   ("not responsible for generating the inner loop code... all buffer ops are
   user-defined callbacks"). ynnpack's "optimize" is kernel selection + graph
   fusion + weight packing, not codegen.
2. **The ynnpack public API is opaque** (verified against `ynnpack/include/
   ynnpack.h` at the pinned commit — see §3).

## 3. What the ynnpack public API exposes — and the constraint it imposes

From the pinned `ynnpack.h`:

- **Graph builders:** ~30 `ynn_define_*` ops — `tensor, iota, unary,
  unary_polynomial, convert(_v2), quantize, dequantize, binary, lut,
  (static_)broadcast, broadcast_like, static_expand_dims, static_reshape,
  fuse_dim(s), split_dim, even_split, concatenate, stack, copy, static_slice,
  slice_like, static_transpose, static_pad, stencil_copy, dot, reduce,
  get_tensor_shape`.
- **Optimize:** `ynn_optimize_subgraph(subgraph, threadpool, flags)` — status
  only. **No introspection** of the chosen tiling/schedule/microkernel.
- **Runtime:** `ynn_create_runtime`, `ynn_reshape_runtime`, `ynn_invoke_runtime`,
  `ynn_set_external_value_data`, `ynn_query_runtime` (only property:
  `ynn_runtime_property_concurrency`).
- **Packing:** none. **Codegen/JIT/LLVM/emit:** none.

**Consequence:** through the public API, ynnpack is a *graph builder + opaque
optimizer + opaque runtime*. It will not tell us which microkernel it picked,
hand us a packed layout, or give us IR. So "get the IR out of ynnpack" means
reaching past the graph API to the microkernel **sources** underneath
(XNNPACK's ukernels), and doing the selection/packing/stitching ourselves or via
a small fork-side hook. That is the subject of §5.

## 4. Target architecture

Keep the front half (detection + the `__ynn_fusion` grouping). Replace the back
half with an emitter that produces a fused LLVM kernel whose inner loops are
ynnpack's microkernels-as-IR, feeding XLA's existing JIT + kernel runtime.

```
HLO fusion (__ynn_fusion)                         ← unchanged front half
  │  ThunkEmitter: if microkernel-codegen enabled, route here ...
  ▼
YnnKernelEmitter : KernelEmitter<LlvmKernelSource>    ← NEW
  │  - EmitKernelPrototype() via KernelApiIrBuilder → XLA_CPU kernel ABI
  │  - emit driver loop nest (M/N/K tiles), partitioned by workgroup_id
  │  - per tile: call the ynnpack microkernel (linked in from bitcode),
  │              then the fused elementwise epilogue on the tile in registers
  │  - link the ukernel IR into the module, inline + specialize
  │  - return KernelDefinition{ KernelSpec(name, NumWorkGroups, buffers),
  │                             LlvmKernelSource(module) }
  ▼
JitCompiler (+ IrCompiler: opt level, target features, fast-math)
  │  → ObjectLoader → CompiledFunctionLibrary (symbol → XLA_CPU_Kernel*)
  ▼
KernelThunk (Thunk::Kind::kKernel)                ← reuses XLA's kernel runtime
  ▼
results
```

Everything from `JitCompiler` down already exists and is what
`DotKernelEmitter`/`ElementalKernelEmitter` use. New surface = `YnnKernelEmitter`
+ the bitcode-extraction pipeline (§5). Anything not covered falls back to
`YnnFusionThunk`, so rollout is incremental and reversible. The kernel ABI
(`KernelApiIrBuilder::EmitKernelPrototype` → `XLA_CPU_KernelCallFrame`,
`workgroup_id`/`num_workgroups`) and the workgroup-grid parallelism (as in
`DotKernelEmitter`'s `NumWorkGroups`) are reused unchanged.

## 5. The core plan: get ynnpack's microkernels out as LLVM IR

XNNPACK microkernels written in C with SIMD intrinsics compile to clean LLVM IR
with clang. The plan extracts them as bitcode at build time and stitches them
into the fused kernel.

### 5.1 The extraction pipeline

1. **In-build bitcode.** A Bazel rule compiles the needed XNNPACK microkernel
   translation units (e.g. the `f32`/`bf16` GEMM ukernels and their pack
   routines) to LLVM **bitcode** using XLA's vendored clang/LLVM toolchain, with
   the target arch flags. The `.bc` is embedded as data in the XLA binary.
   *Generated in-build, never checked in* — so it is always the same LLVM the
   JIT links (see §6, the no-pinning point).
2. **Microkernel registry + selection.** XNNPACK ukernel symbol names are
   regular (e.g. `xnn_f32_gemm_minmax_ukernel_<MR>x<NR>__<isa>`). XLA holds a
   table mapping (dtype, tile `MR×NR`, ISA) → symbol, and selects per fusion
   from `TargetMachineFeatures`. (The public ynn API won't choose for us — §3 —
   so XLA chooses; a fork hook to expose ynnpack's own choice is an optional
   upgrade, §5.3.)
3. **Driver emission.** `YnnKernelEmitter` emits the kernel prototype, the
   M/N/K tiling loops (partitioned into the workgroup grid), the operand
   prologue (pack/stride A; packed weights from step 5/§7.2), and per output
   tile a **call** to the selected ukernel (declared with its ABI: tile dims,
   `a`+stride, packed `w`, `c`+strides, a `params` struct for the built-in
   clamp). Then the elementwise epilogue is emitted on the tile.
4. **Link + specialize.** Load the ukernel (and pack) functions from the
   embedded bitcode into the kernel `llvm::Module` (`llvm::Linker`/`IRMover`),
   mark `alwaysinline`, constant-propagate dims/strides, and run `IrCompiler`'s
   pipeline → inline, unroll, **fuse the epilogue into the ukernel's store**,
   vectorize, emit object.
5. **Packing.** For constant weights, run XNNPACK's pack routine (also available
   as in-build bitcode, or called at compile time) to materialize a packed
   constant buffer once; the ukernel reads it directly (§7.2).
6. **JIT + KernelThunk**, then **fallback** to `YnnFusionThunk` for anything not
   covered.

### 5.2 Option A (XLA emits its own microkernels) — fallback only
The alternative is to have XLA write the GEMM microkernel itself in IR (extend
`tiled_dot_emitter`/`VectorIrBuilder`) using ynnpack only for the high-level
recipe, never XNNPACK's kernels. We keep this **only as a fallback** for tiles
where no usable bitcode ukernel exists (notably asm-only kernels, §5.3), since
on its own it doesn't deliver XNNPACK's tuned performance — the whole point of
"using ynnpack."

### 5.3 Honest gaps (true regardless of approach)
- **Selection.** The public API won't say which ukernel it would use; XLA must
  select (step 2). Faithfully using *ynnpack's* selection needs a small fork
  hook (e.g. a `ynn_query_*` returning the chosen ukernel/tiling per node).
- **Asm-only kernels.** XNNPACK's fastest kernels on some ISAs (much of ARM,
  some x86) are hand-written **assembly**, which cannot be turned into LLVM IR.
  Bitcode extraction covers the intrinsic-C subset only; the rest fall back
  (to Option A or to `YnnFusionThunk`). Coverage is therefore arch-dependent —
  strongest where XNNPACK uses intrinsic-C (much of x86 AVX/AVX-512).
- **Packing-layout coupling.** Using XNNPACK ukernels means using XNNPACK's
  packed-weight layout, so we must use its pack routines (step 5). This couples
  the driver to ukernel/pack versions in the pinned XNNPACK.

## 6. LLVM version: no pinning if generated in-build

Concern (from v2): LLVM IR/bitcode and intrinsics are not stable across major
versions, and a separately-built bitcode artifact could drift from the LLVM XLA
links. **Resolution:** generate the microkernel bitcode *as part of the XLA
build* with XLA's vendored clang (§5.1 step 1) and do **not** check in `.bc`.
Then the embedded IR is, by construction, the exact LLVM version the JIT uses on
every build — there is nothing to pin. (Pinning would only return if we shipped
prebuilt bitcode produced outside the XLA build.)

## 7. Supporting design points

### 7.1 Epilogue / prologue fusion (the main payoff)
`__ynn_fusion` already groups the contraction with its elementwise neighbors in
one HLO computation. Once the ukernel is inlined IR, emit the elementwise chain
on the output tile via `CpuElementalIrEmitter` generators (as
`ElementalKernelEmitter` does) before the store; LLVM fuses it into the
ukernel's epilogue. This removes the materialization the slinky pipeline pays
for between contraction and elementwise. (XNNPACK's ukernel `params` clamp
covers relu/relu6-style activations natively; richer epilogues are emitted by
XLA.)

### 7.2 Packing
No public ynnpack packing API (§3), so:
- **Constant weights:** prepack at compile time using XNNPACK's pack routine
  (in-build bitcode or a direct compile-time call) into a new constant
  allocation in the ukernel's expected layout. No per-call repack.
- **Non-constant operands:** emit a prologue pack into scratch
  (`KernelSpec.scratch_bytes`), or gate such fusions to `YnnFusionThunk` for the
  first cut.

### 7.3 Selection, gating, fallback
- Reuse `YnnMatcher`/`ynn_support` for *what* fuses, unchanged.
- Add a flag (e.g. `xla_cpu_experimental_ynn_codegen`, or codegen variants of
  `LibraryFusionType`) selecting runtime vs codegen per fusion type; default off.
- Emitter returns `absl::Status`; on any unsupported construct (asm-only
  ukernel, dynamic shape, unhandled type/layout/op) **fall back to
  `YnnFusionThunk`**. Both thunks coexist.

### 7.4 Dynamic shapes
Single fused static kernel assumes compile-time shapes (the common case).
Otherwise emit runtime loop bounds (less unrolling) or fall back initially.

### 7.5 Threading
Drop `SlinkyThreadPool`/`ynn_threadpool` from the hot path; parallelism is the
`KernelThunk` workgroup launch over the Eigen pool. The driver picks a workgroup
partition over output tiles (mirrors slinky's parallel dim). Keep the slinky
shim only for fallback fusions.

## 8. Phasing

1. **Skeleton + fallback.** Bitcode rule for one `f32` GEMM ukernel + its pack;
   `YnnKernelEmitter` that links + drives it for a simple `f32` matmul (no
   epilogue) behind a default-off flag; route in `ThunkEmitter`; everything else
   falls back. Land JIT→`KernelThunk` + numerics tests.
2. **Epilogue fusion.** Fold elementwise neighbors onto the tile; add bf16.
3. **Packing.** Compile-time prepack of constant weights.
4. **Coverage.** More tile shapes/ISAs; conv (stencil_copy + dot); evaluate
   asm-kernel gaps; Option-A fallback where needed.
5. **Fidelity/autotuning.** Optional fork hook to use ynnpack's own selection;
   expose driver tiles to `LlvmKernelAutotuner`.

## 9. Risks / open questions

- **Asm-only ukernels (§5.3).** Hard cap on bitcode coverage, arch-dependent;
  mitigated by fallback but limits the win on those ISAs.
- **Selection without a fork hook (§5.3).** XLA-side selection may diverge from
  ynnpack's; a fork `ynn_query_*` closes the gap if we want exact parity.
- **Packing-layout coupling (§5.3/§7.2).** Driver tied to XNNPACK pack/ukernel
  versions in the pinned dep.
- **Build complexity.** Compiling the right XNNPACK TUs to bitcode (include
  paths, `-D` configs, per-arch intrinsics) is the main new build machinery.
- **Two paths to maintain** until codegen reaches parity; flag + fallback keep
  it reversible.
- *(Resolved)* **LLVM pinning** — eliminated by in-build bitcode (§6).

## 10. Decisions needed

1. **Selection ownership:** XLA-side table (no fork change) to start, vs a small
   ynnpack fork hook to expose its own ukernel/tiling choice.
2. **First target ISA + ukernel** for the Phase-1 slice (suggest x86 AVX-512
   `f32` GEMM, where XNNPACK is intrinsic-C and bitcode extraction is cleanest).
3. **Scope of this task:** design only, or land the Phase-1 slice (full
   build/benchmark happens in a complete build env, not this session).
