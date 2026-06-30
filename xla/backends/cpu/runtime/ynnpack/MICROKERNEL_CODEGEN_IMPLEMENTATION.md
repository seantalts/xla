# Implementation spec: ynnpack dot microkernel codegen (Phase 0–2)

Companion to `MICROKERNEL_CODEGEN_DESIGN.md`. A step-by-step build plan for a
narrow slice, grounded in the **actual ynnpack source** at the pinned commit
(`google/XNNPACK@76de138…`, `ynnpack/` tree). Signatures quoted here are real
(see the Appendix). `CONFIRM:` = verify against XLA source while implementing;
`READ:` = read from the ynnpack source (`ynnpack/kernels/`, `ynnpack/subgraph/`).

> ynnpack only. Use ynnpack's own kernels (`ynnpack/kernels/`) and APIs; do
> **not** reference classic XNNPACK microkernels (`src/…`, `xnn_*_ukernel`).

## 0. Conventions
- Paths relative to repo root (`/home/user/xla`). ynnpack paths are within the
  `@XNNPACK` bazel repo (e.g. `@XNNPACK//ynnpack/kernels/dot:dot`).
- Every new path is **flag-gated (default off)** and **falls back** to
  `YnnFusionThunk` when out of slice. Never regress the runtime path.

## 1. Scope of the slice
Support exactly this, fall back otherwise:
- one `dot` inside a `__ynn_fusion` (the `LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT`
  path already produces these);
- `f32`; rank-2, non-batched, canonical `(M,K)x(K,N)->(M,N)`;
- RHS (weights) constant; AVX-512 target; **non-transpose-A** kernel
  (avoid `dot_flag::transpose_a`); no activation epilogue (bias allowed); static
  shapes.

Milestones:
- **Phase 0** — `YnnKernelEmitter` end-to-end with a *naive XLA-emitted* matmul
  (no ynnpack). Validates emitter→JIT→`KernelThunk`. Fully specified; no ynnpack
  internals.
- **Phase 1** — inline the ynnpack `ynnpack/kernels/dot` kernel as bitcode into
  an XLA-emitted driver; constant B packed at compile time via `ynn::packer`;
  bias via `c_in`.
- **Phase 2** — fuse the activation epilogue; tiers; fork hook; `schedule_dot`.

## 2. Files to create / modify
| Action | Path | Purpose |
|---|---|---|
| add | `xla/backends/cpu/codegen/ynn/ynn_kernel_emitter.{h,cc}` | `KernelEmitter<LlvmKernelSource>` |
| add | `xla/backends/cpu/codegen/ynn/BUILD` | targets |
| add (P1) | `xla/backends/cpu/codegen/ynn/kernels/ynn_dot_f32_avx512.cc` | wrapper TU compiled to bitcode |
| modify | `xla/service/cpu/thunk_emitter.{h,cc}` | dispatch + `EmitYnnKernelThunk` |
| modify | `xla/backends/cpu/ynn_support.{h,cc}` | `IsYnnCodegenEligible` |
| modify | `xla/xla.proto` + `xla/debug_options_flags.cc` | `xla_cpu_experimental_ynn_codegen` |
| add | `xla/backends/cpu/codegen/ynn/ynn_kernel_emitter_test.cc` | tests |
| (P1) modify ynnpack fork | `ynnpack/kernels/dot/dot.h` + macro | add `const char* name` to `dot_kernel` (the "fork hook") |

## 3. Step A — flag
`xla.proto` (next free field in `DebugOptions`):
```proto
bool xla_cpu_experimental_ynn_codegen = <NEXT_FREE>;
```
`debug_options_flags.cc`: default `false`; register a `bool_setter_for` mirroring
an existing `xla_cpu_experimental_*` bool. `CONFIRM:` field number.

## 4. Step B — eligibility (`ynn_support.{h,cc}`)
```cpp
bool IsYnnCodegenEligible(const HloInstruction* fusion,
                          const TargetMachineFeatures& target);
```
1. `kFusion`, backend-config kind `== kYnnFusionKind`.
2. fused root is `kDot`, sole compute op.
3. operands+result `F32`, rank 2, canonical (`GetDotCanonicalDims` from
   `xla/backends/cpu/runtime/dot_dims.h`; `lhs_canonical && rhs_canonical`, no
   batch).
4. RHS fusion operand constant (mirror `IsConstant` in `EmitYnnFusionThunk`).
5. `target.has_avx512f()` (`CONFIRM:` accessor).

## 5. Step C — Phase 0 emitter (naive IR)
Mirror `DotKernelEmitter` (`xla/backends/cpu/codegen/dot/dot_kernel_emitter.{h,cc}`).
Header: `YnnKernelEmitter : KernelEmitter<LlvmKernelSource>` with
`(const HloInstruction* fusion, const BufferAssignment*, const TargetMachineFeatures*)`.
`EmitKernelDefinition()`:
```cpp
auto ctx = std::make_unique<llvm::LLVMContext>();
KernelApiIrBuilder kb(*ctx, KernelApiIrBuilder::Options::FromHloModuleConfig(
                                fusion_->GetModule()->config()));
auto module = KernelApiIrBuilder::CreateModule(
    absl::StrCat(fusion_->name(), "_ynn_module"), *ctx);
ASSIGN_OR_RETURN(KernelApiIrBuilder::KernelPrototype proto,
                 kb.EmitKernelPrototype(*module, fusion_, buffer_assignment_,
                                        name(), "_kernel"));
llvm::IRBuilder<> b(*ctx);
b.SetInsertPoint(proto.function->getEntryBlock().getTerminator());
// Phase 0: naive triple loop over proto.arguments[0]/[1] -> proto.results[0]
// (llvm_ir::IrArray EmitReadArrayElement/EmitWriteArrayElement), partitioned by
// proto.workgroup_id.x. Copy idioms from elemental_kernel_emitter.cc /
// dot_op_emitter.cc (kNaiveLlvmIr).
DotOpWorkGroupDim wg = EmitNaiveGemm(b, proto, M, N, K);
LlvmKernelSource source(std::move(ctx), std::move(module));
KernelSpec::Buffers args, results;
for (auto& p : proto.argument_buffers) args.push_back({p.slice, p.shape});
for (auto& p : proto.result_buffers) results.push_back({p.slice, p.shape});
KernelSpec spec(proto.function->getName(), NumWorkGroups{wg.x, wg.y},
                std::move(args), std::move(results),
                std::move(proto.invariant_arguments));
return KernelDefinition(std::move(spec), std::move(source));
```
BUILD: copy deps from `xla/backends/cpu/codegen/dot/BUILD`'s `dot_kernel_emitter`.

## 6. Step D — dispatch + thunk (`thunk_emitter.cc`)
Replace the existing `kYnnFusionKind` branch with codegen-first + fallback:
```cpp
if (backend_config.fusion_config().kind() == kYnnFusionKind) {
  if (hlo_module_config_.debug_options().xla_cpu_experimental_ynn_codegen() &&
      IsYnnCodegenEligible(instruction, target_machine_features_)) {
    absl::StatusOr<ThunkSequence> seq = EmitYnnKernelThunk(instruction);
    if (seq.ok()) return seq;
    VLOG(1) << "ynn codegen fallback: " << seq.status();
  }
  return EmitYnnFusionThunk(instruction);
}
```
`EmitYnnKernelThunk` (mirror `EmitDotThunk`'s codegen branch):
```cpp
YnnKernelEmitter emitter(instruction, &buffer_assignment_,
                         &target_machine_features_);
ASSIGN_OR_RETURN(KernelDefinition kd, emitter.EmitKernelDefinition());
auto spec = kd.spec();
auto src = std::move(kd).TakeSource();
kernels_.push_back({spec.name(), std::move(src).thread_safe_module()});
return MakeKernelThunkSequence(instruction, std::move(spec), MinAlign());
```
Modules flow automatically: `kernels_` → `ConsumeKernels()` → `cpu_compiler.cc`
→ `JitCompiler::AddModule` → `Compile`. **Checkpoint:** Phase 0 numerics pass.

## 7. Step E — Phase 1: inline the ynnpack dot kernel

### 7.1 Fork hook — expose the kernel symbol name
`get_dot_kernel` returns a fn pointer; codegen needs the **symbol name** to emit
a direct, inlinable call. Patch the ynnpack fork: add `const char* name = nullptr;`
to `struct dot_kernel` (`ynnpack/kernels/dot/dot.h`) and populate it from the
`YNN_DOT_KERNEL(arch, name, …)` macro (stringify `name`) and the generator. Tiny,
local. (Until merged, `READ:` the chosen symbol from a locally-generated
`x86_avx512_fp32.inc`.)

### 7.2 Select the kernel at compile time
In the emitter, link host ynnpack and call (real signature, Appendix):
```cpp
ynn::dot_type type{/*a=*/ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};  // CONFIRM enum spelling
ynn::dot_shape shape; shape.m = M; shape.n = N;  // k dims optional
uint64_t arch = MapTargetToYnnArchFlags(*target_machine_);  // -> arch_flag::avx512f|...
ynn::dot_kernel k = ynn::get_dot_kernel(type, shape, /*packed=*/nullptr,
    /*required_flags=*/0, /*transpose_a=*/std::make_optional(false), arch);
// k.name (fork hook), k.block_m, k.block_n, k.block_k, k.tile_n, k.tile_k
```
`MapTargetToYnnArchFlags`: OR together `ynn::arch_flag::{avx512f,avx512bw,
avx512vl,avx512dq,fma3,avx2,…}` from `TargetMachineFeatures`. `CONFIRM:` the
`ynn_type` enum spelling in `ynnpack/include/ynnpack.h`.

### 7.3 Pack constant B at compile time
B is `(K,N)` f32. Pack with `ynn::packer` using the kernel's tile params
(`subgraph/dot.cc` sets `packed_shape{block_n=k.block_n, tile_k=k.tile_k}`):
```cpp
ynn::packer pk(/*transpose=*/false, /*elem_size_bits=*/32,
               /*tile_m=*/k.tile_k, /*tile_n=*/k.block_n);   // READ: confirm tile_m/tile_n mapping for B in subgraph pack node
std::vector<float> packed(PackedSizeFloats(K, N, k.tile_k, k.block_n));  // READ: size formula (round up to tiles)
pk.pack(/*m=*/K, /*n=*/N, /*input_stride=*/N*4, b_data,
        /*output_stride=*/…, /*output_block_stride=*/…, packed.data());  // READ: strides from subgraph pack node
```
Bake `packed` as a private `llvm::GlobalVariable` (ConstantDataArray); keep its
`ptr`. (Non-constant B → fall back in Phase 1.)

### 7.4 Extract the dot kernel as bitcode
Wrapper TU `xla/backends/cpu/codegen/ynn/kernels/ynn_dot_f32_avx512.cc` that
pulls in the AVX-512 f32 dot kernel(s):
```cpp
#define YNN_ARCH_X86_AVX512 1            // CONFIRM the arch-select macro name
#include "ynnpack/kernels/dot/dot.cc"    // includes kernels.inc -> generated .inc
```
`cc_ir_header` target:
```python
load("//xla/codegen/intrinsic/cpp:cc_to_llvm_ir.bzl", "cc_ir_header")
cc_ir_header(
    name = "ynn_dot_f32_avx512_ll",
    src = "kernels/ynn_dot_f32_avx512.cc",
    namespace = "ynn_kernels",
    base_name = "ynn_dot_f32_avx512",
    copts = ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq",
             "-mprefer-vector-width=512"],   # CONFIRM: match ynn_kernel_copts
    deps = [
        "@XNNPACK//ynnpack/kernels/dot:x86_fp32",   # generated srcs; CONFIRM label
        "@XNNPACK//ynnpack/base",                    # CONFIRM
    ],
)
```
Generates `inline const std::string kYnnDotF32Avx512Ir` in `ynn_kernels`.
**Main build risk:** wiring ynnpack's *generated* sources (the `ynn_generate_srcs`
outputs feeding `kernels.inc`) into this compile. Get one TU emitting non-empty
bitcode in isolation first. `CONFIRM:` visibility — ynnpack targets may need a
visibility grant to XLA.

### 7.5 Emit the driver and inline the kernel
Replace `EmitNaiveGemm` with `EmitYnnDotDriver`:
1. Declare the kernel with the exact ABI (Appendix `dot_kernel_fn`):
   `void(i64×8, ptr, i64×3, ptr, i64, ptr, i64, ptr)` — i.e. `(m,n,k3,k2,k1,
   a_stride_m,a_stride_k3,a_stride_k2, a, b_stride_k3,b_stride_k2,b_stride_k1, b,
   c_in_stride_m, c_in, c_out_stride_m, c_out)`. Use `module->getOrInsertFunction(
   k.name, fnty)`.
2. Pointers from the prototype: `a` = `proto.arguments[0]` base; `c_out` =
   `proto.results[0]` base; `b` = packed-B global (§7.3); `c_in` = bias buffer or
   null/zero. `CONFIRM:` `IrArray` base-pointer accessor.
3. Emit a **simple M-blocked loop** (correctness first; `schedule_dot` cache
   tiling is a Phase-2 perf add): partition M over `proto.workgroup_id.x`; loop
   `mb` step `k.block_m`, call the kernel with `m=min(block_m, M-mb)`, `n=N`,
   `k3=k2=1, k1=K` (`CONFIRM:` K-split convention; if `K % tile_k`, handle the
   tail as `subgraph/dot.cc` does with a padded-A `alloca`), strides:
   `a_stride_m=K*4`, `b_stride_*` per the packed layout (§7.3),
   `c_*_stride_m=N*4`. `NumWorkGroups.x = ceil_div(ceil_div(M,block_m), …)`.
4. Bias: pass the bias buffer as `c_in` (`C_out = C_in + A·B`); if no bias, pass a
   zero buffer or a kernel variant with `c_in=null` (`READ:` whether null `c_in`
   is allowed; `subgraph/dot.cc` chains `c_in` for split-K).
5. Link + inline:
   ```cpp
   xla::codegen::CppGenIntrinsicLibrary lib(ynn_kernels::kYnnDotF32Avx512Ir,
                                            "ynn_dot");
   lib.LinkIntoModule(*module);
   xla::codegen::GetCppGenFunction(module, k.name);  // InternalLinkage+AlwaysInline
   ```
   `IrCompiler` inlines the kernel, constant-folds dims, emits the object.

**Checkpoint (Phase 1):** the JIT'd kernel inlines the ynnpack dot kernel;
numerics match; FileCheck sees `call @<k.name>` pre-inline.

### 7.6 Phase 2 (optimization, later)
- Activation epilogue: emit the elementwise neighbors on the `c_out` tile (ynnpack
  `kernels/elementwise` inlined, or `CpuElementalIrEmitter`) before store; LLVM
  fuses into the kernel epilogue.
- Use `ynn::schedule_dot` + `ynn::run_dot` (header-only) for cache-aware tiling.
- bf16 (`x86_avx512bf16_*`), AVX2/FMA3 tiers (runtime-selected like
  `GetCppGenIrString`), `dot_flag::transpose_a` kernels (with A transpose).

## 8. BUILD deps
`ynn_kernel_emitter` deps += `@XNNPACK//ynnpack/kernels/dot:dot` (host
`get_dot_kernel`/`packer`), `:ynn_dot_f32_avx512_ll`,
`//xla/codegen/intrinsic/cpp:_cpp_gen_intrinsics`. `CONFIRM:` ynnpack target
visibility to XLA.

## 9. Tests (`ynn_kernel_emitter_test.cc`)
1. **Numerics:** one `f32` constant-RHS matmul; flag on vs off; aligned and
   unaligned `M,N,K` (tail). Add a biased case.
2. **FileCheck/IR** (mirror `tanh_test`/`eigen_unary_test`): pre-opt module has
   `call @<k.name>` + the packed-B global; post-opt it's inlined.
3. **Fallback:** bf16 / non-constant RHS / transpose dot with flag on still runs
   via `YnnFusionThunk`, matches reference.
4. **Benchmark:** extend `benchmarks/ynn_fusion_benchmark_test.cc`.

## 10. Known simplifications → later
| Shortcut | Why OK | Later |
|---|---|---|
| packed B baked as LLVM global | correct, simple | constant buffer via HLO rewrite; drop unused RHS arg |
| simple M-block loop | correct | `schedule_dot`/`run_dot` cache tiling |
| RHS constant only | enables compile-time pack | prologue pack node for dynamic B |
| no activation | bias via `c_in` only | P2 epilogue fusion |
| one kernel/tier hardcoded | proves path | fork hook + AVX2/FMA3 tiers, runtime select |
| `f32` only | smallest slice | bf16; int8 via `dequantize_dot` |
| non-transpose only | avoids A relayout | `dot_flag::transpose_a` kernels |

## 11. Fallback discipline
`IsYnnCodegenEligible` is the gate; false → `YnnFusionThunk`. Emitter non-OK →
VLOG(1) + fall back. Flag default `false`.

---

## Appendix — verbatim ynnpack signatures (pinned commit)

`ynnpack/kernels/dot/dot.h`:
```cpp
typedef void (*dot_kernel_fn)(size_t m, size_t n, size_t k3, size_t k2,
    size_t k1, size_t a_stride_m, size_t a_stride_k3, size_t a_stride_k2,
    const void* a, size_t b_stride_k3, size_t b_stride_k2, size_t b_stride_k1,
    const void* b, size_t c_in_stride_m, const void* c_in,
    size_t c_out_stride_m, void* c_out);
// C_out(i,j) = C_in(i,j) + sum_{k3,k2,k1} A(i,k3,k2,k1) * B(k3,k2,k1,j)

struct dot_kernel { dot_kernel_fn kernel; int block_m, block_n, block_k,
                    tile_n, tile_k; uint32_t flags; float cost; };
struct dot_type { ynn_type a, b, c; };
struct dot_shape { std::optional<size_t> m, n, k1, k2, k3; };
struct dot_packed_shape { int block_n, tile_k; };

dot_kernel get_dot_kernel(const dot_type& type, const dot_shape& shape = {},
    const dot_packed_shape* dot_packed_shape = nullptr,
    uint32_t required_flags = 0,
    std::optional<bool> transpose_a = std::nullopt,
    uint64_t arch_flags = get_supported_arch_flags());

namespace dot_flag { enum { transpose_a=1<<0, consistent_arithmetic=1<<1,
                            unaligned_b=1<<2 }; }
```
`ynnpack/kernels/dot/pack.h`:
```cpp
class packer {
 public:
  packer(bool transpose, size_t elem_size_bits, size_t tile_m, size_t tile_n);
  void pack(size_t m, size_t n, size_t input_stride, const void* input,
            size_t output_stride, size_t output_block_stride, void* output);
};
// packed(mi,ni,mo,no) = input(mo*tile_m+mi, no*tile_n+ni), zero-padded to tiles
```
`ynnpack/kernels/dot/schedule.h` (header-only): `schedule_dot(...)` →
`span<dot_loop>`; `run_dot(loops, m, n, ks, block_m, block_n, block_k, …, DotFn)`;
`block_dot_{m,n,k}(...)`. `ynnpack/base/arch.h`: `arch_flag::{fma3=1<<6,
avx512f=1<<7, avx512bw, avx512vl, avx512dq, avx512bf16, avx512vnni, amxbf16,
amxint8, sme, sme2, …}`, `uint64_t get_supported_arch_flags()`.

Reference integration: `ynnpack/subgraph/dot.cc` (`make_dot_impl`) — sets
`packed_shape{block_n, tile_k}`, calls `get_dot_kernel`, wraps `kernel.kernel` in
a lambda, drives via `schedule_dot`+`run_dot`, bias via `c_in`, handles the
`k % tile_k` tail with a padded-A `alloca`.
```
