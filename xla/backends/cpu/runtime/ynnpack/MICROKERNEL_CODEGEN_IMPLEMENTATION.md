# Implementation spec: ynnpack dot microkernel codegen (Phase 0–2)

Companion to `MICROKERNEL_CODEGEN_DESIGN.md` (read that for rationale). A
step-by-step build plan for a narrow vertical slice. Items marked `CONFIRM:` are
facts to verify against source while implementing. Items marked `READ:` must be
read from the **ynnpack checkout** (`ynnpack/kernels/`, `ynnpack/subgraph/`) —
they could not be verified remotely and must not be guessed.

> ynnpack only. Use ynnpack's own kernels in `ynnpack/kernels/` and its subgraph
> API. Do **not** reference classic XNNPACK microkernels (`src/…`,
> `xnn_*_ukernel`, `xnn_pack_*`); ynnpack is a rewrite and does not use them.

## 0. Conventions
- Paths are relative to the repo root (`/home/user/xla`).
- Every new code path is **flag-gated (default off)** and **falls back** to the
  existing `YnnFusionThunk` when the input isn't in the slice. Never regress the
  runtime path.

## 1. Scope of the vertical slice
Support exactly this, fall back otherwise:
- A single `dot` already wrapped in a `__ynn_fusion` (the existing
  `LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT` path produces these).
- `f32`; rank-2, non-batched, canonical `(M,K) x (K,N) -> (M,N)`.
- RHS (weights) is a compile-time constant.
- Target CPU has AVX-512.
- No elementwise epilogue (plain dot); static shapes.

Three milestones (each independently testable):
- **Phase 0** — `YnnKernelEmitter` wired end-to-end emitting a *naive* XLA LLVM-
  IR matmul (no ynnpack kernel). Proves the emitter → JIT → `KernelThunk` seam,
  flag, dispatch, workgroup partition, numerics. **Fully specified here; depends
  on nothing ynnpack-internal.**
- **Phase 1** — replace the inner loop with ynnpack's `ynnpack/kernels/dot`
  kernel, pulled in as bitcode via `cc_ir_header` and inlined; constant weights
  packed at compile time via ynnpack's packing node. **Structure specified;
  exact ynnpack ABI is a `READ:` step.**
- **Phase 2** — epilogue fusion (later).

## 2. Files to create / modify

| Action | Path | Purpose |
|---|---|---|
| add | `xla/backends/cpu/codegen/ynn/ynn_kernel_emitter.{h,cc}` | the new `KernelEmitter<LlvmKernelSource>` |
| add | `xla/backends/cpu/codegen/ynn/BUILD` | build target |
| add (P1) | `xla/backends/cpu/codegen/ynn/kernels/` + `cc_ir_header` targets | embedded ynnpack-kernel bitcode |
| modify | `xla/service/cpu/thunk_emitter.{h,cc}` | dispatch + `EmitYnnKernelThunk` |
| modify | `xla/backends/cpu/ynn_support.{h,cc}` | `IsYnnCodegenEligible(...)` predicate |
| modify | `xla/xla.proto` + `xla/debug_options_flags.cc` | `xla_cpu_experimental_ynn_codegen` flag |
| add | `xla/backends/cpu/codegen/ynn/ynn_kernel_emitter_test.cc` | numerics + FileCheck tests |

## 3. Step A — the flag
`xla/xla.proto`, CPU section of `DebugOptions` (use next free field number):
```proto
// Emit JIT'd LLVM-IR ynnpack kernels for __ynn_fusion instead of the ynnpack
// runtime. Experimental; falls back to YnnFusionThunk when ineligible.
bool xla_cpu_experimental_ynn_codegen = <NEXT_FREE>;
```
`debug_options_flags.cc`: default `false`; register a `bool_setter_for` entry
mirroring an existing `xla_cpu_experimental_*` bool. `CONFIRM:` next free field.

## 4. Step B — eligibility predicate
`ynn_support.h/.cc`:
```cpp
bool IsYnnCodegenEligible(const HloInstruction* fusion,
                          const TargetMachineFeatures& target);
```
Checklist:
1. `kFusion` with backend-config kind `== kYnnFusionKind`.
2. Fused root is `kDot` and the only compute op.
3. dot operands+result `F32`, rank 2, canonical dims (reuse
   `GetDotCanonicalDims` from `xla/backends/cpu/runtime/dot_dims.h`; require
   `lhs_canonical && rhs_canonical`, no batch).
4. RHS fusion operand is constant (mirror `IsConstant(operand)` in
   `ThunkEmitter::EmitYnnFusionThunk`).
5. `target.has_avx512f()` (`CONFIRM:` accessor in `target_machine_features.h`).
Return false otherwise.

## 5. Step C — the emitter (Phase 0: naive IR)
Mirror `DotKernelEmitter`
(`xla/backends/cpu/codegen/dot/dot_kernel_emitter.{h,cc}`).

`ynn_kernel_emitter.h`:
```cpp
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/llvm_kernel_source.h"
namespace xla::cpu {
class YnnKernelEmitter final : public KernelEmitter<LlvmKernelSource> {
 public:
  YnnKernelEmitter(const HloInstruction* fusion,
                   const BufferAssignment* buffer_assignment,
                   const TargetMachineFeatures* target_machine);
  absl::string_view name() const final { return "ynn_kernel_emitter"; }
  absl::StatusOr<KernelDefinition> EmitKernelDefinition() override;
 private:
  const HloInstruction* fusion_;
  const BufferAssignment* buffer_assignment_;
  const TargetMachineFeatures* target_machine_;
};
}  // namespace xla::cpu
```

`ynn_kernel_emitter.cc` — `EmitKernelDefinition()` (mirrors the verified
`DotKernelEmitter::EmitKernelDefinition`):
```cpp
absl::StatusOr<YnnKernelEmitter::KernelDefinition>
YnnKernelEmitter::EmitKernelDefinition() {
  const HloModule* hlo_module = fusion_->GetModule();
  auto ctx = std::make_unique<llvm::LLVMContext>();
  KernelApiIrBuilder kernel_api_ir_builder(
      *ctx, KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));
  std::unique_ptr<llvm::Module> module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(fusion_->name(), "_ynn_module"), *ctx);

  ASSIGN_OR_RETURN(KernelApiIrBuilder::KernelPrototype proto,
                   kernel_api_ir_builder.EmitKernelPrototype(
                       *module, fusion_, buffer_assignment_, name(), "_kernel"));

  llvm::IRBuilder<> b(*ctx);
  b.SetInsertPoint(proto.function->getEntryBlock().getTerminator());

  const Shape& lhs = fusion_->operand(0)->shape();   // CONFIRM operand order
  const Shape& out = fusion_->shape();
  const int64_t M = lhs.dimensions(0), K = lhs.dimensions(1);
  const int64_t N = out.dimensions(1);

  // Phase 0: naive triple loop over [m-band by workgroup_id.x, n, k] using
  // proto.arguments[0]/[1] and proto.results[0] (llvm_ir::IrArray) with
  // EmitReadArrayElement/EmitWriteArrayElement. Copy idioms from
  // elemental_kernel_emitter.cc and dot_op_emitter.cc (kNaiveLlvmIr strategy).
  DotOpWorkGroupDim wg = EmitNaiveGemm(b, proto, M, N, K);  // helper you write

  LlvmKernelSource source(std::move(ctx), std::move(module));
  KernelSpec::Buffers args, results;
  for (const auto& p : proto.argument_buffers) args.push_back({p.slice, p.shape});
  for (const auto& p : proto.result_buffers) results.push_back({p.slice, p.shape});
  KernelSpec spec(proto.function->getName(), NumWorkGroups{wg.x, wg.y},
                  std::move(args), std::move(results),
                  std::move(proto.invariant_arguments));
  return KernelDefinition(std::move(spec), std::move(source));
}
```
`KernelApiIrBuilder`/`KernelPrototype`/`CreateModule`/`EmitKernelPrototype` are
in `xla/backends/cpu/codegen/kernel_api_ir_builder.h`;
`DotOpWorkGroupDim`/`NumWorkGroups` in `dot_kernel_emitter.cc` /
`xla/runtime/work_group.h`.

### Phase-0 BUILD
`cc_library` `ynn_kernel_emitter` with deps mirroring
`xla/backends/cpu/codegen/dot/BUILD`'s `dot_kernel_emitter`
(`//xla/codegen:kernel_emitter`, `:kernel_api_ir_builder`,
`//xla/codegen:llvm_kernel_source`, `//xla/codegen:kernel_spec`,
`//xla/service:buffer_assignment`, `target_machine_features`, llvm `Core`/
`ir_headers`, absl).

## 6. Step D — thunk_emitter dispatch + thunk
Existing dispatch in `thunk_emitter.cc`:
```cpp
if (backend_config.fusion_config().kind() == kYnnFusionKind) {
  return EmitYnnFusionThunk(instruction);
}
```
Try codegen first, fall back:
```cpp
if (backend_config.fusion_config().kind() == kYnnFusionKind) {
  if (hlo_module_config_.debug_options().xla_cpu_experimental_ynn_codegen() &&
      IsYnnCodegenEligible(instruction, target_machine_features_)) {
    absl::StatusOr<ThunkSequence> seq = EmitYnnKernelThunk(instruction);
    if (seq.ok()) return seq;
    VLOG(1) << "ynn codegen failed, falling back: " << seq.status();
  }
  return EmitYnnFusionThunk(instruction);
}
```
Add `EmitYnnKernelThunk` mirroring `EmitDotThunk`'s codegen branch (verified):
```cpp
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitYnnKernelThunk(
    const HloInstruction* instruction) {
  YnnKernelEmitter emitter(instruction, &buffer_assignment_,
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
Declare in `thunk_emitter.h`. The module then flows automatically: `kernels_` →
`ConsumeKernels()` → `cpu_compiler.cc` → `JitCompiler::AddModule` → `Compile`.
Add the BUILD dep on `//xla/backends/cpu/codegen/ynn:ynn_kernel_emitter`.

**Checkpoint (Phase 0):** flag on, `f32` matmul runs through the new kernel and
matches reference. Land tests (§9).

## 7. Step E — Phase 1: inline the ynnpack dot kernel

### 7.1 READ the ynnpack dot kernel ABI (do this first)
From the ynnpack checkout, read `ynnpack/kernels/dot/` (and the "headers
describing the kernels" the README mentions) and record:
- the kernel source file(s) and the **exact generated function symbol** for an
  `f32`, AVX-512 dot kernel;
- its **exact ABI** — `READ:` it may be a tile microkernel taking raw
  pointers + strides + tile extents, **or** a slinky-style buffer-callback taking
  buffer descriptors (slinky invokes kernels as buffer callbacks). The driver in
  §7.4 must construct exactly what this ABI expects.
- its blocking params (tile M/N/K, vector width) and whether it is intrinsic-C++
  (expected) or asm (if asm, use the extern-call fallback, design §5.1).

### 7.2 Embed the kernel as bitcode
Add a `cc_ir_header` target (`xla/codegen/intrinsic/cpp/cc_to_llvm_ir.bzl`)
compiling that ynnpack kernel TU:
```python
load("//xla/codegen/intrinsic/cpp:cc_to_llvm_ir.bzl", "cc_ir_header")
cc_ir_header(
    name = "ynn_dot_f32_avx512_ll",
    src = "kernels/ynn_dot_f32_avx512.cc",   # wrapper #include-ing the ynnpack TU
    namespace = "ynn_kernels",
    base_name = "ynn_dot_f32_avx512",
    deps = ["@XNNPACK//ynnpack:ynnpack_h"],  # CONFIRM: ynnpack kernel headers dep
    copts = ["-mavx512f", "-mprefer-vector-width=512"],
)
```
Generates `inline const std::string kYnnDotF32Avx512Ir` in `ynn_kernels`.
`CONFIRM:` the include paths / config the ynnpack kernel TU needs (it is
generated and may include ynnpack-internal headers; the cleanest is a tiny
wrapper TU that `#include`s the generated kernel with the right config). Get one
TU to emit non-empty bitcode in isolation before integrating.

### 7.3 Pack weights at compile time (ynnpack packing node)
Packing in ynnpack is a subgraph node, not a public call. `READ:`
`ynnpack/subgraph/` for the pack node and which `ynnpack/kernels/` kernel it
uses. Then at emit time, for the constant RHS:
1. read the RHS constant `Literal` (`const float*`, shape `(K,N)`);
2. run the ynnpack pack routine (extracted as a host call, since ynnpack is
   linked into the XLA binary, or replicated) into a `std::vector<float>` of the
   packed size `READ:` from ynnpack;
3. bake it as a private LLVM global constant in the module
   (`llvm::ConstantDataArray::get` → `GlobalVariable`, `PrivateLinkage`); keep
   its `ptr`.
(For non-constant weights → fall back in Phase 1.)

### 7.4 Emit the driver and inline the kernel
Replace `EmitNaiveGemm` with `EmitYnnDotDriver`:
1. Declare the ynnpack kernel in the module with the ABI from §7.1
   (`module->getOrInsertFunction(symbol, fnty)`).
2. Compute pointers/args that ABI expects: LHS base = `proto.arguments[0]`
   base ptr; out base = `proto.results[0]` base ptr; packed weights = the global
   from §7.3; plus any strides/extents/params/descriptors the ABI requires.
   `CONFIRM:` how to get a raw base `ptr` from `llvm_ir::IrArray`.
3. Partition the output over `proto.workgroup_id.x` (one tile-band per workgroup)
   and emit the per-tile `call` to the kernel; set `NumWorkGroups.x` accordingly.
4. After emitting the body, link + inline the kernel:
   ```cpp
   xla::codegen::CppGenIntrinsicLibrary lib(
       ynn_kernels::kYnnDotF32Avx512Ir, "ynn_dot");
   lib.LinkIntoModule(*module);
   xla::codegen::GetCppGenFunction(module, "<ynnpack kernel symbol>");
   ```
   `IrCompiler` then inlines the kernel, constant-folds dims, emits the object.
   (`CppGenIntrinsicLibrary`/`GetCppGenFunction`:
   `xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h`.)

Note: the RHS still appears in `proto.arguments[1]` but is unused (we read the
baked global). Correct but wasteful; §10 removes it.

**Checkpoint (Phase 1):** the JIT'd kernel inlines the ynnpack dot kernel;
numerics match; the FileCheck test sees the kernel call pre-inline.

## 8. Step F — BUILD deps for Phase 1
Add `:ynn_dot_f32_avx512_ll` and
`//xla/codegen/intrinsic/cpp:_cpp_gen_intrinsics` to `ynn_kernel_emitter` deps.

## 9. Tests
1. **Numerics** (`ynn_kernel_emitter_test.cc`): one `f32` constant-RHS matmul;
   flag on vs off; compare. Cover `M,N,K` aligned and unaligned to the tile.
2. **FileCheck/IR** (mirror `tanh_test`/`eigen_unary_test`): assert the pre-opt
   module contains the ynnpack-kernel `call` + baked packed-weights global;
   optionally assert it is inlined away post-opt.
3. **Fallback**: a non-eligible dot (bf16 / non-constant RHS) with the flag on
   still runs (via `YnnFusionThunk`) and matches reference.
4. **Benchmark**: extend
   `xla/backends/cpu/benchmarks/ynn_fusion_benchmark_test.cc`.

## 10. Known simplifications and later fixes
| Slice shortcut | Why OK now | Later phase |
|---|---|---|
| packed weights baked as LLVM global | correct; simple | pack into a real constant buffer (HLO rewrite); drop the unused RHS arg |
| RHS must be constant | enables compile-time pack | prologue pack node for non-constant weights |
| no epilogue | plain dot only | P2: fuse elementwise neighbors (ynnpack unary/binary kernels inlined, or `ElementalKernelEmitter` generators) |
| one ynnpack kernel/tile hardcoded | proves path | ynnpack fork hook returns kernel id + blocking + pack node (design §5.4) |
| one ISA tier (AVX-512) | cleanest extraction | add AVX2/256 tier, runtime-select like `GetCppGenIrString` |
| `f32` only | smallest slice | bf16; later quantized (`ynnpack/kernels/dequantize_dot`) |
| naive loop nest | single dot only | reproduce more of slinky's fusion/tiling for fused subgraphs |
| asm kernel (if any) | rare in ynnpack | extern-call fallback on the same driver (design §5.1) |

## 11. Fallback discipline (every phase)
- `IsYnnCodegenEligible` is the single gate; false → `YnnFusionThunk`.
- `EmitYnnKernelThunk` non-OK → VLOG(1) + fall back.
- Flag default `false`; no behavior change unless enabled.
```
