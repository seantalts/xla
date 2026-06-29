# Design: LLVM-IR microkernel codegen for the XLA:CPU ynnpack path

Status: draft / RFC (v5)
Scope: XLA:CPU backend, `__ynn_fusion` path

> Terminology: **ynnpack** is a from-scratch successor to XNNPACK (a "spiritual
> successor", per its README), living in the `ynnpack/` tree of the
> `google/XNNPACK` repo. It has **its own** kernels in `ynnpack/kernels/`
> (generated C++ with SIMD intrinsics), a subgraph API (`ynnpack/include/
> ynnpack.h`), and uses **slinky** for loop fusion/scheduling. It does **not**
> reuse classic XNNPACK microkernels (`src/…`, `xnn_*_ukernel`), and neither
> does this design — everything below is ynnpack-only.

## 1. Goal

Today XLA:CPU offloads matched fusions (dot / conv / reduce / elementwise) to
ynnpack and lets ynnpack *execute* them at runtime (subgraph → slinky loops →
ynnpack kernels). This proposal changes the back half: **emit the ynnpack
kernels as LLVM IR** into a fused driver kernel that XLA JIT-compiles, and run it
with a plain `KernelThunk` — instead of handing the subgraph to ynnpack's
runtime. The extract→embed→link→inline machinery already exists in-tree
(`xla/codegen/intrinsic/cpp/`, used for `tanh` and Eigen unary ops); ynnpack
kernels are generated intrinsic-C++, so they lift to bitcode through it.

The win is not "LLVM beats ynnpack's kernels" — we *keep* ynnpack's kernels, now
as IR. The win is what the offload model hides from the compiler once the kernel
is IR in our module:

- **Epilogue/prologue fusion** — fold surrounding elementwise ops (ynnpack's
  unary/binary kernels, or XLA's elemental emitter) onto each output tile *in
  registers* before the store, instead of separate slinky pipeline stages over
  materialized buffers.
- **Shape/target specialization** — the kernel inlines into a driver with
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
  │    ynn_invoke_runtime()  ← slinky fuses/schedules the loops and calls
  │                            ynnpack kernels, parallel via SlinkyThreadPool
  ▼
results
```

Two facts shape this design:

1. **ynnpack does not emit code; slinky is an interpreter.** ynnpack ships
   generated intrinsic-C++ kernels in `ynnpack/kernels/`; slinky is a runtime
   that fuses/schedules loops across subgraph nodes and calls those kernels as
   callbacks. ynnpack's "optimize" is graph optimization + kernel selection +
   inserting nodes (e.g. **packing is a subgraph node**), not codegen. Per the
   README, "most of the work that the operator API in XNNPACK performed has been
   moved into subgraph nodes, e.g. packing weights is handled by a subgraph
   node."
2. **The ynnpack public API is opaque** (verified against
   `ynnpack/include/ynnpack.h` at the pinned commit — see §3).

## 3. What the ynnpack public API exposes — and the constraint it imposes

From the pinned `ynnpack/include/ynnpack.h`:

- **Graph builders:** ~30 `ynn_define_*` ops (`tensor, dot, reduce, unary,
  binary, convert, quantize, dequantize, lut, stencil_copy, static_pad,
  static_transpose, broadcast, concatenate, ...`).
- **Optimize:** `ynn_optimize_subgraph(subgraph, threadpool, flags)` — status
  only. **No introspection** of the chosen schedule/kernel/tiling.
- **Runtime:** `ynn_create_runtime`, `ynn_reshape_runtime`, `ynn_invoke_runtime`,
  `ynn_set_external_value_data`, `ynn_query_runtime` (only property:
  `ynn_runtime_property_concurrency`).
- **Packing:** none (it's a subgraph node, internal). **Codegen/JIT/LLVM:** none.

**Consequence:** the public API will not tell us which kernel/tiling it picked or
give us IR. "Get the IR out of ynnpack" therefore means reaching to the kernel
**sources** in `ynnpack/kernels/` and emitting the loop nest ourselves (the role
slinky plays at runtime), with a small **fork-side hook** to recover ynnpack's
own kernel/schedule selection (decided in §5.4).

## 4. Target architecture

Keep the front half (detection + the `__ynn_fusion` grouping). Replace the back
half with an emitter that produces a fused LLVM kernel whose inner loops are
ynnpack's kernels-as-IR, feeding XLA's existing JIT + kernel runtime.

```
HLO fusion (__ynn_fusion)                         ← unchanged front half
  │  ThunkEmitter: if microkernel-codegen enabled, route here ...
  ▼
YnnKernelEmitter : KernelEmitter<LlvmKernelSource>    ← NEW
  │  - EmitKernelPrototype() via KernelApiIrBuilder → XLA_CPU kernel ABI
  │  - emit the loop nest (the role slinky plays at runtime), partitioned by
  │    workgroup_id
  │  - per tile: call the ynnpack kernel(s) (linked in from embedded bitcode),
  │    then the fused elementwise epilogue in registers
  │  - link kernel IR via CppGenIntrinsicLibrary, inline + specialize
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
+ the ynnpack-kernel bitcode targets (§5). Anything not covered falls back to
`YnnFusionThunk`, so rollout is incremental and reversible. The kernel ABI
(`KernelApiIrBuilder::EmitKernelPrototype`) and workgroup-grid parallelism
(`NumWorkGroups`, as in `DotKernelEmitter`) are reused unchanged.

The genuinely new responsibility is **emitting the loop nest that slinky
computes at runtime** (bounds, tiling, the order kernels are called). For a
single dot this is a simple M/N/K nest; general fused subgraphs would need to
reproduce more of slinky's fusion/tiling, which is why the slice (impl doc)
starts with one dot and falls back otherwise.

## 5. The core plan: extract ynnpack kernels as IR and stitch them

### 5.1 Reuse the existing extract→embed→link machinery
XLA already compiles C++ to embedded LLVM IR and links it into JIT modules for
intrinsics. ynnpack kernels are generated intrinsic-C++, so the same path works:

- **Build:** `cc_ir_header` (`xla/codegen/intrinsic/cpp/cc_to_llvm_ir.bzl`)
  compiles a `.cc`/`.c` with `-emit-llvm -O3 -mprefer-vector-width=… [-m<isa>]`,
  extracts `.llvmbc`, and embeds it as `inline const std::string k<Name>Ir` via
  the `embed_bitcode` tool + `//xla/util:embedded_constant_buffers`. We add
  `cc_ir_header` targets for the ynnpack kernel TUs from `ynnpack/kernels/`
  (e.g. `ynnpack/kernels/dot/…`). **Generated in-build, never checked in.**
- **Consume:** `CppGenIntrinsicLibrary::LinkIntoModule(dst)` parses the embedded
  bitcode, sets the host datalayout, links into the kernel module, and marks each
  linked def `InternalLinkage + AlwaysInline`. `GetCppGenFunction(module, name)`
  fetches a kernel for inlining; empty-bitcode guards fall back. (All in
  `xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.{h,cc}`.)
- **Specialize:** `IrCompiler`'s pipeline inlines the kernel into the driver,
  constant-folds dims, unrolls, **fuses the epilogue into the kernel's store**,
  vectorizes, emits the object.

Because ynnpack kernels are generated intrinsic-C++ (not hand asm), the
"can't lift assembly" problem that dogs classic XNNPACK is largely absent here.
For any kernel that is *not* available as liftable IR, an **extern call** to a
precompiled ynnpack kernel from the same driver is a fallback tier (XLA already
does this shape for the `kEigen` dot strategy): it keeps the optimized driver +
`KernelThunk` runtime but, being an opaque call, gives no epilogue fusion or
constant-prop. Inline-asm would only elide the (amortized, negligible) call and
stays equally opaque, so it is not worth it.

### 5.2 The variant-count question: bounded ISA tiers, runtime-selected
For Eigen unary, XLA ships **two** tiers (`eigen_unary_32_ll` 256-bit,
`eigen_unary_64_ll` AVX-512) and `GetCppGenIrString(options)` picks one **at
runtime from the JIT target's features**. We do the same: embed a **small set of
ISA tiers** of the ynnpack kernel (not a blob per micro-arch) and select by
`TargetMachineFeatures`. The compiled kernel contains only the selected one (the
linker pulls only referenced symbols). For AOT, embed only the target tier.

### 5.3 Open decision: which ynnpack kernel form to feed `cc_ir_header`?
ynnpack kernels are generated; pick the granularity:
- **(I) ISA-specialized intrinsic kernels** (the tuned `ynnpack/kernels/dot`
  variants) → one bitcode per ISA tier; preserves the tuning; variant count =
  tiers × kernels.
- **(P) A portable/scalar generated variant** → 1–2 tiers, lean on LLVM to
  vectorize; bounded but less tuned.
Recommendation: **(I)** for x86 AVX-512 `f32` dot first.

### 5.4 Selection: a ynnpack fork hook (decided)
The public API won't reveal ynnpack's choice (§3). Add a small hook to the
ynnpack fork that, given a node + target, returns the chosen kernel **identity**
— the `ynnpack/kernels/` symbol, its tile/blocking params, required ISA tier,
and the associated packing-node kernel. XLA maps the ISA tier → embedded bitcode
variant (à la `GetCppGenIrString`) and links that symbol.

### 5.5 Option A (XLA emits its own microkernel) — fallback only
Have XLA write the dot microkernel itself in IR (extend `tiled_dot_emitter`/
`VectorIrBuilder`). Kept only as a fallback where a ynnpack kernel can't be
extracted, since alone it doesn't deliver ynnpack's tuned performance.

### 5.6 Honest gaps / things to read from ynnpack source
These were **not** verifiable remotely; the implementer must read them from the
ynnpack checkout (`ynnpack/kernels/`, `ynnpack/subgraph/`):
- **Kernel ABI.** ynnpack kernels may take raw pointer+stride+tile-dim args, or
  slinky-style buffer descriptors (slinky calls kernels as buffer callbacks).
  This determines what the driver must construct per call. Read
  `ynnpack/kernels/dot/` (and the "headers describing the kernels" the README
  mentions) to get the exact signature and symbol naming.
- **Packing.** Packing is a subgraph node, not a public API. For constant
  weights, replicate/extract that node's kernel to prepack at compile time; read
  `ynnpack/subgraph/` for the pack node and which `ynnpack/kernels/` kernel it
  uses.
- **Loop structure.** XLA must emit the loop nest slinky would compute. Trivial
  for one dot; broader subgraphs need more of slinky's fusion/tiling logic.

## 6. LLVM version: no pinning, by construction
`cc_ir_header` compiles with the build's own clang/LLVM and embeds the result;
`LinkIntoModule` links it into the JIT module built by the same LLVM. Since the
bitcode is regenerated every build (never checked in), the embedded IR always
matches the LLVM the JIT links — nothing to pin. The Eigen/tanh precedent works
exactly this way today.

## 7. Supporting design points

### 7.1 Epilogue / prologue fusion (the main payoff)
`__ynn_fusion` already groups the contraction with its elementwise neighbors in
one HLO computation. Once the dot kernel is inlined IR, emit the elementwise
chain on the output tile (via ynnpack's unary/binary kernels inlined the same
way, or XLA's `CpuElementalIrEmitter` generators) before the store; LLVM fuses
it into the kernel's epilogue.

### 7.2 Packing (a ynnpack subgraph node)
Packing weights is a ynnpack subgraph node, so there is no public pack call.
- **Constant weights:** run the packing node's kernel at compile time (extracted
  as bitcode or replicated) to materialize a packed constant; the dot kernel
  reads it directly. No per-call repack.
- **Non-constant operands:** emit a prologue pack stage, or gate to
  `YnnFusionThunk` first.

### 7.3 Selection, gating, fallback
- Reuse `YnnMatcher`/`ynn_support` for *what* fuses, unchanged.
- Add a flag (e.g. `xla_cpu_experimental_ynn_codegen`) selecting runtime vs
  codegen per fusion type; default off.
- Emitter returns `absl::Status`; on any unsupported construct **fall back to
  `YnnFusionThunk`**. Both thunks coexist.

### 7.4 Dynamic shapes
Single fused static kernel assumes compile-time shapes (the common case).
Otherwise emit runtime loop bounds (less unrolling) or fall back initially.

### 7.5 Threading
Drop `SlinkyThreadPool` from the hot path; parallelism is the `KernelThunk`
workgroup launch over the Eigen pool. The driver picks a workgroup partition over
output tiles (mirrors slinky's parallel dim). Keep the slinky shim only for
fallback fusions.

## 8. Phasing

1. **Skeleton + fallback.** `YnnKernelEmitter` wired end-to-end on a naive XLA-
   emitted `f32` matmul (no ynnpack kernel yet) behind a default-off flag;
   route in `ThunkEmitter`; everything else falls back. Validates the
   emitter→JIT→`KernelThunk` seam.
2. **ynnpack dot kernel.** Extract the `ynnpack/kernels/dot` intrinsic-C++ kernel
   for x86 AVX-512 as bitcode; inline it into the driver; constant-weight pack at
   compile time.
3. **Epilogue fusion.** Fold elementwise neighbors onto the tile; add bf16.
4. **Selection hook + tiers.** Add the ynnpack fork hook (§5.4); add an AVX2/256
   tier; runtime tier selection.
5. **Coverage / fidelity.** reduce/conv via the corresponding `ynnpack/kernels/`;
   non-constant packing; expose tiles to `LlvmKernelAutotuner`.

## 9. Risks / open questions

- **ynnpack kernel ABI + loop structure (§5.6).** Must be read from source;
  shapes the driver. Biggest unknown.
- **Reproducing slinky's schedule.** Easy for one dot; harder for fused
  subgraphs — bounds the near-term scope.
- **Fork hook maintenance (§5.4).**
- **Packing (§7.2).** Replicating/extracting the pack node correctly.
- **Build complexity.** Getting ynnpack kernel TUs to emit clean bitcode through
  `cc_ir_header` (include paths, generated-source config).
- *(Resolved)* **LLVM pinning** — eliminated by in-build bitcode (§6).
- *(Largely moot for ynnpack)* **asm lifting** — kernels are generated
  intrinsic-C++; extern-call is the fallback for any non-liftable kernel (§5.1).

## 10. Decisions needed

1. **Kernel form (§5.3):** ISA-specialized intrinsic ynnpack kernels (tuned,
   more variants) vs a portable variant (bounded, less tuned). Rec: intrinsic,
   x86 AVX-512 `f32` dot first.
2. **First target tier + op** for Phase 2 (rec: x86 AVX-512 `f32` dot).
3. **Scope:** design only, or land the Phase-1 slice (build/benchmark in a full
   env, not this session).

*Decided:* runtime model = `KernelThunk` (not slinky); selection = ynnpack fork
hook (§5.4); microkernel source = ynnpack's own kernels (`ynnpack/kernels/`),
**not** classic XNNPACK.
