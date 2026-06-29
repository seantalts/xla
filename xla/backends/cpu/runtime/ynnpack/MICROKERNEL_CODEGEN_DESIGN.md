# Design: LLVM-IR microkernel codegen for the XLA:CPU ynnpack path

Status: draft / RFC (v4)
Scope: XLA:CPU backend, `__ynn_fusion` path

## 1. Goal

Today XLA:CPU offloads matched fusions (dot / conv / reduce / elementwise) to
**ynnpack** — the next-gen graph API inside the XNNPACK fork — and lets ynnpack
*execute* them at runtime. This proposal changes the back half: **get
ynnpack's microkernels out as LLVM IR**, embed that IR into a fused kernel that
XLA JIT-compiles, and run it with a plain `KernelThunk` — instead of handing the
subgraph to ynnpack's runtime to execute. The extract→embed→link→inline
machinery already exists in-tree (`xla/codegen/intrinsic/cpp/`, used for `tanh`
and Eigen unary ops); this reuses it.

The win is not "LLVM beats XNNPACK's microkernels" — we *keep* XNNPACK's
microkernels, now as IR. The win is what the offload model hides from the
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

- **Graph builders:** ~30 `ynn_define_*` ops (`tensor, dot, reduce, unary,
  binary, convert, quantize, dequantize, lut, stencil_copy, static_pad,
  static_transpose, broadcast, concatenate, ...`).
- **Optimize:** `ynn_optimize_subgraph(subgraph, threadpool, flags)` — status
  only. **No introspection** of the chosen tiling/schedule/microkernel.
- **Runtime:** `ynn_create_runtime`, `ynn_reshape_runtime`, `ynn_invoke_runtime`,
  `ynn_set_external_value_data`, `ynn_query_runtime` (only property:
  `ynn_runtime_property_concurrency`).
- **Packing:** none. **Codegen/JIT/LLVM/emit:** none.

**Consequence:** the public API will not tell us which microkernel it picked,
hand us a packed layout, or give us IR. "Get the IR out of ynnpack" therefore
means reaching to the microkernel **sources** underneath (XNNPACK's ukernels)
and doing selection/packing/stitching ourselves — with a small **fork-side hook**
to recover ynnpack's own selection (decided in §5.4).

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
  │  - per tile: call the ynnpack microkernel (linked in from embedded
  │              bitcode), then the fused elementwise epilogue in registers
  │  - link ukernel IR via CppGenIntrinsicLibrary, inline + specialize
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
+ the bitcode targets (§5). Anything not covered falls back to `YnnFusionThunk`,
so rollout is incremental and reversible. The kernel ABI
(`KernelApiIrBuilder::EmitKernelPrototype`) and workgroup-grid parallelism
(`NumWorkGroups`, as in `DotKernelEmitter`) are reused unchanged.

## 5. The core plan: extract XNNPACK ukernels as IR and stitch them

### 5.1 Reuse the existing extract→embed→link machinery
XLA already compiles C++ to embedded LLVM IR and links it into JIT modules for
intrinsics. We reuse it wholesale:

- **Build:** `cc_ir_header` (`xla/codegen/intrinsic/cpp/cc_to_llvm_ir.bzl`)
  compiles a `.cc` with `-emit-llvm -O3 -mprefer-vector-width=… [-m<isa>]`,
  extracts the `.llvmbc` section, and embeds it as `inline const std::string
  k<Name>Ir` via the `embed_bitcode` tool + `//xla/util:embedded_constant_buffers`.
  We add `cc_ir_header` targets for the XNNPACK GEMM ukernel(s) + pack routine(s)
  we want. **Generated in-build, never checked in.**
- **Consume:** `CppGenIntrinsicLibrary::LinkIntoModule(dst)` parses the embedded
  bitcode, sets the host datalayout, `llvm::Linker::linkModules` into the kernel
  module, and marks each linked def `InternalLinkage + AlwaysInline`.
  `GetCppGenFunction(module, name)` fetches a ukernel for inlining;
  `AreEigenIntrinsicsAvailable()`-style guards detect empty bitcode and fall
  back. (All in `xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.{h,cc}`.)
- **Specialize:** `IrCompiler`'s pass pipeline inlines the ukernel into the
  driver, constant-folds the dims, unrolls K, **fuses the epilogue into the
  ukernel's store**, vectorizes, emits the object.

### 5.2 The variant-count question (the "!!!!"): bounded ISA tiers, runtime-selected
This is a solved pattern in the precedent. For Eigen unary, XLA ships **two**
tiers — `eigen_unary_32_ll` (256-bit) and `eigen_unary_64_ll` (AVX-512) — and
`GetCppGenIrString(options)` picks one **at runtime from the JIT target's
features** (`+avx512f` & prefer-width 512 → the 512 blob, else the 256 blob).

We do the same: embed a **small set of ISA tiers** (not a blob per micro-arch)
and select by `TargetMachineFeatures`. Binary cost is a few blobs; the *compiled
kernel* contains only the one selected ukernel (the linker pulls only referenced
symbols). For AOT, embed only the target tier.

### 5.3 The real open decision: what C do we feed `cc_ir_header`?
The granularity of the embedded source trades performance against
variant-count / coverage:

- **(P) Portable XNNPACK reference kernels** → 1–2 tiers; lean on LLVM's
  auto-vectorizer + `-mprefer-vector-width` + the JIT target to specialize.
  Bounded and clean, but you mostly get LLVM's codegen, **not** XNNPACK's
  hand-tuning. (This is what the Eigen precedent does — effectively Option A
  seeded from XNNPACK's C.)
- **(I) ISA-specific intrinsic XNNPACK kernels** → one bitcode per ISA tier you
  support; preserves hand-tuning for covered tiers; **asm-only kernels
  excluded**; variant count = tiers × kernels embedded.

Recommendation: start with **(I)** for **x86 AVX-512 `f32` GEMM** (XNNPACK is
intrinsic-C there, so extraction is clean and the tuning is preserved), and use
**(P)** / Option-A fallback where intrinsic kernels don't exist or are asm.

### 5.4 Selection: a ynnpack fork hook (decided)
Because the public API won't tell us its choice (§3), add a small hook to the
XNNPACK fork that, given a node + target, returns the chosen ukernel **identity**
— symbol name, tile `MR×NR`, required ISA tier, and the matching pack routine.
XLA maps the ISA tier → embedded bitcode variant (à la `GetCppGenIrString`) and
links that symbol. This keeps XLA's emitted kernel faithful to ynnpack's own
selection, at the cost of a fork patch to maintain.

### 5.5 Option A (XLA emits its own microkernels) — fallback only
Have XLA write the GEMM microkernel itself in IR (extend
`tiled_dot_emitter`/`VectorIrBuilder`), using ynnpack only for the recipe. Kept
**only as a fallback** for tiles with no usable bitcode ukernel (asm-only, §5.6),
since alone it doesn't deliver XNNPACK's tuned performance.

### 5.6 Honest gaps
- **Asm-only ukernels.** XNNPACK's fastest kernels on many ISAs (much of ARM,
  some x86) are hand-written **assembly**, which cannot become LLVM IR. Lifting
  machine code (llvm-mctoll, RetDec, remill) yields un-reoptimizable IR that
  defeats fusion — not viable. Coverage via bitcode is the **intrinsic-C
  subset** (strong on x86 AVX/AVX-512; partial on ARM NEON; asm kernels fall
  back). Coverage is therefore arch-dependent.
- **Packed-layout coupling.** Using XNNPACK ukernels means using its packed
  layout and pack routines (§7.2); the driver is coupled to ukernel/pack
  versions in the pinned XNNPACK.
- **Runtime model.** We can't ship clang, so C→IR compilation is build-time
  only; "runtime" specialization is the JIT linking + optimizing the embedded
  IR for the host. No per-host recompilation of sources.

## 6. LLVM version: no pinning, by construction
`cc_ir_header` compiles with the build's own clang/LLVM and embeds the result;
`LinkIntoModule` links it into the JIT module built by the same LLVM. Since the
bitcode is regenerated every build (never checked in), the embedded IR always
matches the LLVM the JIT links — there is nothing to pin. The Eigen/tanh
precedent works exactly this way today.

## 7. Supporting design points

### 7.1 Epilogue / prologue fusion (the main payoff)
`__ynn_fusion` already groups the contraction with its elementwise neighbors in
one HLO computation. Once the ukernel is inlined IR, emit the elementwise chain
on the output tile via `CpuElementalIrEmitter` generators (as
`ElementalKernelEmitter` does) before the store; LLVM fuses it into the
ukernel's epilogue. XNNPACK's ukernel `params` clamp covers relu/relu6-style
activations natively; richer epilogues are emitted by XLA.

### 7.2 Packing
No public ynnpack packing API (§3), so:
- **Constant weights:** prepack at compile time using XNNPACK's pack routine
  (also pulled in as `cc_ir_header` bitcode, or called at compile time) into a
  new constant allocation in the ukernel's expected layout. No per-call repack.
- **Non-constant operands:** emit a prologue pack into scratch
  (`KernelSpec.scratch_bytes`), or gate such fusions to `YnnFusionThunk` first.

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

1. **Skeleton + fallback.** `cc_ir_header` target for one x86 AVX-512 `f32` GEMM
   ukernel + its pack routine; `YnnKernelEmitter` that links them via
   `CppGenIntrinsicLibrary` and drives a simple `f32` matmul (no epilogue) behind
   a default-off flag; route in `ThunkEmitter`; everything else falls back. Land
   JIT→`KernelThunk` + numerics tests.
2. **Epilogue fusion.** Fold elementwise neighbors onto the tile; add bf16.
3. **Packing.** Compile-time prepack of constant weights.
4. **Selection hook + tiers.** Add the ynnpack fork hook (§5.4); add an AVX2/256
   tier; runtime tier selection.
5. **Coverage / fidelity.** conv (stencil_copy + dot); ARM NEON intrinsic
   kernels; Option-A fallback where asm-only; expose tiles to
   `LlvmKernelAutotuner`.

## 9. Risks / open questions

- **Granularity tradeoff (§5.3).** Portable-C is bounded but modest perf;
  intrinsic kernels preserve tuning but multiply variants and exclude asm.
- **Asm-only ukernels (§5.6).** Hard, arch-dependent cap on bitcode coverage;
  mitigated by fallback but limits the win on ARM especially.
- **Fork hook maintenance (§5.4).** A patch to the XNNPACK fork to expose
  selection (+ pack identity).
- **Packed-layout coupling (§5.6/§7.2).**
- **Build complexity.** Picking the right XNNPACK TUs and `-D`/`-m` flags so
  they compile to clean bitcode through `cc_ir_header`.
- *(Resolved)* **LLVM pinning** — eliminated by in-build bitcode (§6); the
  tanh/Eigen precedent proves it.

## 10. Decisions needed

1. **Source granularity (§5.3):** portable XNNPACK C (bounded, modest perf) vs
   ISA-specific intrinsic kernels (tuned, more variants, asm excluded).
   Recommendation: intrinsic kernels for x86 AVX-512 `f32` GEMM first.
2. **First target tier + kernel** for Phase 1 (recommend x86 AVX-512 `f32` GEMM).
3. **Scope:** design only, or land the Phase-1 slice (build/benchmark happens in
   a full env, not this session).

*Decided:* runtime model = `KernelThunk` (not slinky); selection = ynnpack fork
hook (§5.4).
