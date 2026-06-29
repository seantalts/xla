# Implementation spec: ynnpack GEMM microkernel codegen (Phase 0–1)

Companion to `MICROKERNEL_CODEGEN_DESIGN.md` (read that for rationale). This doc
is a step-by-step build plan concrete enough to implement directly. It targets a
narrow vertical slice and lists every file, signature, and seam. Items marked
`CONFIRM:` are small facts to verify against source while implementing.

## 0. Conventions
- All paths are relative to the repo root (`/home/user/xla`).
- "the slice" = the Phase-0/1 scope in §1.
- Every new code path is **gated by a default-off flag** and **falls back** to
  the existing `YnnFusionThunk` when the input isn't in the slice. Correctness
  first; never regress the runtime path.

## 1. Scope of the vertical slice
Support exactly this, fall back otherwise:
- A single `dot` already wrapped in a `__ynn_fusion` (the existing
  `LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT` path produces these).
- `f32` only; rank-2, non-batched, canonical dot `(M,K) x (K,N) -> (M,N)`
  (no transpose, no batch dims).
- RHS (weights) is a **compile-time constant**.
- Target CPU has **AVX-512** (`TargetMachineFeatures::has_avx512f()` or the
  feature string contains `+avx512f`).
- No elementwise epilogue (plain dot).
- Static shapes.

Two milestones:
- **Phase 0** — wire a new `YnnGemmKernelEmitter` end-to-end emitting a *naive*
  LLVM-IR matmul (no XNNPACK). Proves the emitter → JIT → `KernelThunk` seam,
  the flag, dispatch, workgroup partitioning, and numerics.
- **Phase 1** — replace the inner loop with an XNNPACK **intrinsic-C** GEMM
  ukernel, pulled in as bitcode via `cc_ir_header` and inlined; weights packed
  at compile time and baked as an LLVM global constant.

## 2. Files to create / modify

| Action | Path | Purpose |
|---|---|---|
| add | `xla/backends/cpu/codegen/ynn/ynn_gemm_kernel_emitter.{h,cc}` | the new `KernelEmitter<LlvmKernelSource>` |
| add | `xla/backends/cpu/codegen/ynn/BUILD` | build target for the emitter |
| add (P1) | `xla/backends/cpu/codegen/ynn/ukernels/` + `cc_ir_header` targets | embedded ukernel bitcode |
| modify | `xla/service/cpu/thunk_emitter.{h,cc}` | dispatch + `EmitYnnGemmKernelThunk` |
| modify | `xla/backends/cpu/ynn_support.{h,cc}` | `IsYnnGemmCodegenEligible(...)` predicate |
| modify | `xla/xla.proto` + `xla/debug_options_flags.cc` | `xla_cpu_experimental_ynn_gemm_codegen` flag |
| add | `xla/backends/cpu/codegen/ynn/ynn_gemm_kernel_emitter_test.cc` | numerics + FileCheck tests |

## 3. Step A — the flag
In `xla/xla.proto`, in the CPU section of `DebugOptions`, add (use the next free
field number):
```proto
// Emit JIT'd LLVM-IR GEMM microkernels for __ynn_fusion dots instead of the
// ynnpack runtime. Experimental; falls back to YnnFusionThunk when ineligible.
bool xla_cpu_experimental_ynn_gemm_codegen = <NEXT_FREE>;
```
In `xla/debug_options_flags.cc`: default it `false` in `DefaultDebugOptions()`
and register a `bool_setter_for(...)` flag entry mirroring an existing
`xla_cpu_experimental_*` bool. `CONFIRM:` next free field number in the proto.

## 4. Step B — eligibility predicate
In `ynn_support.h/.cc` add:
```cpp
// True iff `fusion` is a __ynn_fusion wrapping a single dot that the GEMM
// microkernel codegen slice supports (f32, rank-2, canonical, constant RHS,
// AVX-512 target).
bool IsYnnGemmCodegenEligible(const HloInstruction* fusion,
                              const TargetMachineFeatures& target);
```
Implementation checklist:
1. `fusion->opcode()==kFusion` and backend-config kind `== kYnnFusionKind`.
2. The fused computation's root is a `kDot` and the only "compute" op
   (`MakeInstructionPostOrder()` is [param, param, dot] modulo bitcasts).
3. `dot` operands and result are `F32`, rank 2; `DotDimensionNumbers` are the
   canonical `lhs={contracting:1}, rhs={contracting:0}`, no batch dims. Reuse
   `GetDotCanonicalDims` from `xla/backends/cpu/runtime/dot_dims.h` and require
   `lhs_canonical && rhs_canonical`.
4. The RHS operand of the fusion is a constant (mirror the `IsConstant(operand)`
   check used in `ThunkEmitter::EmitYnnFusionThunk`).
5. `target.has_avx512f()` (`CONFIRM:` exact accessor in
   `target_machine_features.h`; else parse `get_target_feature_string()`).
Return false on anything else.

## 5. Step C — the emitter (Phase 0: naive IR)
Mirror `DotKernelEmitter` exactly (see
`xla/backends/cpu/codegen/dot/dot_kernel_emitter.{h,cc}`).

`ynn_gemm_kernel_emitter.h`:
```cpp
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/llvm_kernel_source.h"
namespace xla::cpu {
class YnnGemmKernelEmitter final
    : public KernelEmitter<LlvmKernelSource> {
 public:
  YnnGemmKernelEmitter(const HloInstruction* fusion,
                       const BufferAssignment* buffer_assignment,
                       const TargetMachineFeatures* target_machine);
  absl::string_view name() const final { return "ynn_gemm_kernel_emitter"; }
  absl::StatusOr<KernelDefinition> EmitKernelDefinition() override;
 private:
  const HloInstruction* fusion_;
  const BufferAssignment* buffer_assignment_;
  const TargetMachineFeatures* target_machine_;
};
}  // namespace xla::cpu
```

`ynn_gemm_kernel_emitter.cc` — `EmitKernelDefinition()` skeleton (mirrors the
verified `DotKernelEmitter::EmitKernelDefinition` body):
```cpp
absl::StatusOr<YnnGemmKernelEmitter::KernelDefinition>
YnnGemmKernelEmitter::EmitKernelDefinition() {
  const HloModule* hlo_module = fusion_->GetModule();
  auto ctx = std::make_unique<llvm::LLVMContext>();

  KernelApiIrBuilder kernel_api_ir_builder(
      *ctx,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));
  std::unique_ptr<llvm::Module> module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(fusion_->name(), "_ynn_gemm_module"), *ctx);

  // Args derived from fusion operands: arguments[0]=LHS, arguments[1]=RHS;
  // results[0]=out. (KernelApiIrBuilder reads BufferAssignment.)
  ASSIGN_OR_RETURN(
      KernelApiIrBuilder::KernelPrototype proto,
      kernel_api_ir_builder.EmitKernelPrototype(
          *module, fusion_, buffer_assignment_, name(), "_kernel"));

  llvm::IRBuilder<> b(*ctx);
  b.SetInsertPoint(proto.function->getEntryBlock().getTerminator());

  // Shapes (canonical f32 (M,K)x(K,N)).
  const Shape& lhs = fusion_->operand(0)->shape();   // CONFIRM operand order
  const Shape& out = fusion_->shape();
  const int64_t M = lhs.dimensions(0), K = lhs.dimensions(1);
  const int64_t N = out.dimensions(1);

  // Workgroup partition: split M into proto.num_workgroups.x bands.
  // Phase 0: emit a naive triple loop over [m in this band, n, k] using
  // IrArray element accessors:
  //   acc = 0; for k: acc += lhs[m,k] * rhs[k,n]; out[m,n] = acc;
  // Use llvm_ir::IrArray::Index + EmitReadArrayElement/EmitWriteArrayElement
  // with proto.arguments[0], proto.arguments[1], proto.results[0].
  // Partition with proto.workgroup_id.x like DotKernelEmitter does.
  DotOpWorkGroupDim wg = EmitNaiveGemm(b, proto, M, N, K);  // helper you write

  // Package.
  LlvmKernelSource source(std::move(ctx), std::move(module));
  KernelSpec::Buffers args, results;
  for (const auto& p : proto.argument_buffers) args.push_back({p.slice, p.shape});
  for (const auto& p : proto.result_buffers) results.push_back({p.slice, p.shape});
  KernelSpec spec(proto.function->getName(),
                  NumWorkGroups{wg.x, wg.y},
                  std::move(args), std::move(results),
                  std::move(proto.invariant_arguments));
  return KernelDefinition(std::move(spec), std::move(source));
}
```
Notes:
- `KernelApiIrBuilder`, `KernelPrototype`, `CreateModule`, `EmitKernelPrototype`
  are in `xla/backends/cpu/codegen/kernel_api_ir_builder.h`.
- `DotOpWorkGroupDim`/`NumWorkGroups`: see `dot_kernel_emitter.cc` /
  `xla/runtime/work_group.h`.
- For the naive loop, copy index/loop emission idioms from
  `elemental_kernel_emitter.cc` and `service/cpu/dot_op_emitter.cc`
  (`kNaiveLlvmIr` strategy) — those already emit exactly this.

### Phase-0 BUILD (`xla/backends/cpu/codegen/ynn/BUILD`)
A `cc_library` named `ynn_gemm_kernel_emitter` with deps mirroring the
`dot_kernel_emitter` target's deps (`//xla/codegen:kernel_emitter`,
`:kernel_api_ir_builder`, `//xla/codegen:llvm_kernel_source`,
`//xla/codegen:kernel_spec`, `//xla/service:buffer_assignment`,
`target_machine_features`, llvm `Core`/`ir_headers`, absl). Copy from
`xla/backends/cpu/codegen/dot/BUILD`.

## 6. Step D — thunk_emitter dispatch + thunk
In `thunk_emitter.cc`, the existing dispatch is:
```cpp
if (backend_config.fusion_config().kind() == kYnnFusionKind) {
  return EmitYnnFusionThunk(instruction);
}
```
Change to try codegen first, then fall back:
```cpp
if (backend_config.fusion_config().kind() == kYnnFusionKind) {
  if (hlo_module_config_.debug_options()
          .xla_cpu_experimental_ynn_gemm_codegen() &&
      IsYnnGemmCodegenEligible(instruction, target_machine_features_)) {
    absl::StatusOr<ThunkSequence> seq = EmitYnnGemmKernelThunk(instruction);
    if (seq.ok()) return seq;
    VLOG(1) << "ynn gemm codegen failed, falling back: " << seq.status();
  }
  return EmitYnnFusionThunk(instruction);
}
```
Add `EmitYnnGemmKernelThunk` mirroring `EmitDotThunk`'s codegen branch
(verified pattern):
```cpp
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitYnnGemmKernelThunk(
    const HloInstruction* instruction) {
  YnnGemmKernelEmitter emitter(instruction, &buffer_assignment_,
                               &target_machine_features_);
  ASSIGN_OR_RETURN(KernelDefinition kernel_definition,
                   emitter.EmitKernelDefinition());
  auto kernel_spec = kernel_definition.spec();
  auto kernel_source = std::move(kernel_definition).TakeSource();
  kernels_.push_back(
      {kernel_spec.name(), std::move(kernel_source).thread_safe_module()});
  return MakeKernelThunkSequence(instruction, std::move(kernel_spec),
                                 /*min_alignment=*/MinAlign());
}
```
Declare it in `thunk_emitter.h` next to `EmitDotThunk`. The emitted module then
flows automatically: `kernels_` → `ConsumeKernels()` →
`cpu_compiler.cc` → `JitCompiler::AddModule` → `Compile`. No other wiring.
Add the BUILD dep on `//xla/backends/cpu/codegen/ynn:ynn_gemm_kernel_emitter`.

**Checkpoint (end of Phase 0):** with the flag on, an `f32` constant-RHS matmul
runs through the new kernel and matches the reference. Land tests (§9) here.

## 7. Step E — Phase 1: the XNNPACK ukernel as bitcode

### 7.1 Pick the ukernel
Find an **intrinsic-C** (not `asm_`) f32 GEMM minmax ukernel in the XNNPACK
source, e.g. a file under `src/f32-gemm/gen/` defining a symbol like
`xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast`. Record its tile params:
`MR` (e.g. 7), `NR` (e.g. 16), `KR`, `SR`. `CONFIRM:` exact symbol, file, and
`NR/KR/SR` (for `*_broadcast` typically `KR=1, SR=1`); cross-check against
`xnn_init_f32_gemm_config` in the XNNPACK source.

ABI (verified):
```c
void <ukernel>(size_t mr, size_t nc, size_t kc,
               const float* a, size_t a_stride,
               const float* w, float* c,
               size_t cm_stride, size_t cn_stride,
               const struct xnn_f32_minmax_params* params);
// struct xnn_f32_minmax_params { struct { float scale, min, max; } scalar; };
```

### 7.2 Embed it as bitcode
Add a `cc_ir_header` target (macro in
`xla/codegen/intrinsic/cpp/cc_to_llvm_ir.bzl`) compiling that ukernel TU:
```python
load("//xla/codegen/intrinsic/cpp:cc_to_llvm_ir.bzl", "cc_ir_header")
cc_ir_header(
    name = "f32_gemm_7x16_avx512_ll",
    src = "ukernels/f32-gemm-7x16-minmax-avx512f-broadcast.c",  # copied/wrapped
    namespace = "ynn_ukernels",
    base_name = "f32_gemm_7x16_avx512",
    deps = ["@XNNPACK//:xnnpack_headers_or_microkernel_deps"],  # CONFIRM dep
    copts = ["-mavx512f", "-mprefer-vector-width=512"],
)
```
This generates `inline const std::string kF32Gemm7x16Avx512Ir` in
`ynn_ukernels` namespace. `CONFIRM:` the include paths / `-D` config the ukernel
TU needs (XNNPACK uses internal headers; the cleanest is a tiny wrapper `.c`
that `#include`s the gen file with the right config, compiled with XNNPACK's
include dirs). Get *one* ukernel TU to emit non-empty bitcode in isolation
before integrating.

### 7.3 Pack weights at compile time (host call)
XNNPACK is already linked into the XLA binary (`@XNNPACK`), so call the host pack
function directly in the emitter — no bitcode needed for packing:
```c
void xnn_pack_f32_gemm_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                             size_t kr, size_t sr, const float* kernel,
                             const float* bias, const void* scale,
                             float* packed_weights, size_t extra_bytes,
                             const void* params);
```
In the emitter:
1. Read the RHS constant literal: `fusion_->operand(1)` is the constant; get its
   `Literal` (it's a `kConstant` or has `literal()`); raw data is `const float*`
   of shape `(K, N)` → this is GOI with `g=1, nc=N, kc=K`.
2. Compute packed size in floats (broadcast, `KR=SR=1`):
   `packed_floats = round_up(N, NR) * (K + 1)`  // (1 bias + K weights) per chan
   `CONFIRM:` against XNNPACK; allocate `+ NR` slack to be safe.
3. `std::vector<float> packed(packed_floats, 0.f);`
   `xnn_pack_f32_gemm_goi_w(1, N, K, NR, KR, SR, weights, /*bias=*/nullptr,
                            /*scale=*/nullptr, packed.data(), 0, nullptr);`
   (`bias=nullptr` → zero bias, correct for a plain dot.)
4. Bake `packed` as a private LLVM global constant array in the module
   (`llvm::ConstantDataArray::get(ctx, packed)` →
   `new llvm::GlobalVariable(..., PrivateLinkage, ...)`); keep its `ptr`.

### 7.4 Emit the driver that calls the ukernel
Replace `EmitNaiveGemm` with `EmitXnnGemmDriver`:
1. Declare the ukernel in the module with the exact ABI:
   ```cpp
   llvm::Type* i64 = b.getInt64Ty(); llvm::Type* ptr = b.getPtrTy();
   auto* fty = llvm::FunctionType::get(
       b.getVoidTy(), {i64,i64,i64, ptr,i64, ptr, ptr, i64,i64, ptr},
       /*isVarArg=*/false);
   auto ukernel = module->getOrInsertFunction("<ukernel symbol>", fty);
   ```
2. Build the params global `{scale=1.0f, min=-FLT_MAX, max=FLT_MAX}` (struct of
   3 floats) as a private global; `CONFIRM:` `scale` semantics (likely 1.0).
3. `a` = base pointer of `proto.arguments[0]` (LHS); `c` = base pointer of
   `proto.results[0]`; `w` = the packed global ptr from §7.3.
   `CONFIRM:` how to get the raw base `ptr` from an `llvm_ir::IrArray`
   (`.GetBasePointer()` or similar).
4. Strides (bytes): `a_stride = K*4`, `cm_stride = N*4`, `cn_stride = NR*4`.
   `CONFIRM:` `cn_stride` semantics against the ukernel source (stride to advance
   `c` by one NR-block of columns).
5. Partition M over `proto.workgroup_id.x`: each workgroup handles a band of
   rows `[m0, m1)`. Loop `mb` from `m0` to `m1` step `MR`:
   ```
   mr = min(MR, m1 - mb)
   call ukernel(mr, N, K,
                a + mb*K (as ptr, f32 GEP), K*4,
                w, c + mb*N (f32 GEP), N*4, NR*4, params)
   ```
   `NumWorkGroups.x = ceil_div(ceil_div(M, MR), rows_per_wg)`; pick a simple
   partition (e.g. one MR-row-block per workgroup → `x = ceil_div(M, MR)`).
6. After emitting the body, link the ukernel definition and make it inline:
   ```cpp
   xla::codegen::CppGenIntrinsicLibrary lib(
       ynn_ukernels::kF32Gemm7x16Avx512Ir, "ynn_gemm_ukernel");
   lib.LinkIntoModule(*module);                       // adds def, alwaysinline
   xla::codegen::GetCppGenFunction(module, "<ukernel symbol>");  // mark inline
   ```
   `IrCompiler` then inlines the ukernel into the driver, constant-folds dims,
   and emits the object. (`CppGenIntrinsicLibrary`/`GetCppGenFunction` live in
   `xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h`.)

Note: in Phase 1 the LHS/out are real kernel args (good). The RHS arg still
appears in `proto.arguments[1]` but is unused (we read the baked global). That's
correct but wastes a buffer; §10 removes it.

**Checkpoint (end of Phase 1):** the JIT'd kernel calls the inlined XNNPACK
ukernel; numerics match; the FileCheck test sees the ukernel call pre-inline.

## 8. Step F — register the BUILD bitcode dep
Add `:f32_gemm_7x16_avx512_ll` and `//xla/codegen/intrinsic/cpp:_cpp_gen_intrinsics`
to the `ynn_gemm_kernel_emitter` deps so the embedded string + linker helpers are
available.

## 9. Tests
1. **Numerics** (`ynn_gemm_kernel_emitter_test.cc`): build an HLO module with one
   `f32` constant-RHS matmul; run with `xla_cpu_experimental_ynn_gemm_codegen`
   on; compare against the same module with the flag off (reference). Use the CPU
   test harness used by `dot` tests (see
   `xla/backends/cpu/codegen/dot/...test` / `xla/service/cpu/tests`). Cover
   `M,N,K` both multiples of `MR/NR` and not (tail handling).
2. **FileCheck/IR** (mirror `tanh_test`/`eigen_unary_test`): assert the emitted
   module (pre-optimization) contains a `call` to the ukernel symbol and the
   baked packed-weights global; optionally assert it's gone (inlined)
   post-optimization.
3. **Fallback**: a non-eligible dot (e.g. `bf16`, or non-constant RHS) with the
   flag on still runs (via `YnnFusionThunk`) and matches reference.
4. **Benchmark**: extend `xla/backends/cpu/benchmarks/ynn_fusion_benchmark_test.cc`
   to compare codegen vs runtime path.

## 10. Known simplifications and their later fixes
| Slice shortcut | Why OK now | Later phase |
|---|---|---|
| RHS packed + baked as LLVM global | correct; simple | P3: pack into a real constant buffer (HLO rewrite) so it isn't re-embedded per JIT and the unused RHS arg is dropped |
| RHS arg present but unused | correct; minor waste | P3: drop the operand |
| no epilogue | plain dot only | P2: fuse elementwise neighbors onto the output tile (see `ElementalKernelEmitter` generators) |
| single ukernel/tile hardcoded | proves path | P4: ynnpack fork hook returns ukernel id + `MR/NR/KR/SR` + pack fn (design §5.4) |
| one ISA tier (AVX-512) | cleanest extraction | P4: add AVX2/256 tier, select at runtime like `GetCppGenIrString` |
| `f32` only | smallest slice | P5: bf16; later quantized (qc8w) |
| x86 only | intrinsic-C kernels lift cleanly | P5: ARM NEON intrinsic kernels; asm-only → fallback |
| RHS must be constant | enables compile-time pack | P3: prologue/prepack thunk for non-constant weights |

## 11. Fallback discipline (must hold every phase)
- `IsYnnGemmCodegenEligible` is the single gate; anything false → `YnnFusionThunk`.
- `EmitYnnGemmKernelThunk` returning non-OK → log at VLOG(1) and fall back.
- Flag default `false`. No behavior change unless explicitly enabled.
```
