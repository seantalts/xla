# Implementation spec: ynnpack dot microkernel codegen (Phase 0–2)

Companion to `MICROKERNEL_CODEGEN_DESIGN.md`. A step-by-step build plan for a
narrow slice, grounded in the **actual ynnpack source** at the pinned commit
(`google/XNNPACK@76de138…`, `ynnpack/` tree). Signatures quoted here are real
(see the Appendix). All previously open `READ:`/`CONFIRM:` items have been
resolved against the pinned source; where a fact was read from a specific file
it is cited as `(file:approx-line)` so an implementer can re-verify.

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
- **Phase 2** — fuse the activation epilogue; fma3/neon tiers; `schedule_dot`.

## 2. Files to create / modify
| Action | Path | Purpose |
|---|---|---|
| add | `xla/backends/cpu/codegen/ynn/ynn_kernel_emitter.{h,cc}` | `KernelEmitter<LlvmKernelSource>` |
| add | `xla/backends/cpu/codegen/ynn/BUILD` | targets, incl. the `cc_ir_header` bitcode target |
| modify | `xla/service/cpu/thunk_emitter.{h,cc}` | dispatch + `EmitYnnKernelThunk` |
| modify | `xla/backends/cpu/ynn_support.{h,cc}` | `IsYnnCodegenEligible` |
| modify | `xla/xla.proto` + `xla/debug_options_flags.cc` | `xla_cpu_experimental_ynn_codegen` |
| add | `xla/backends/cpu/codegen/ynn/ynn_kernel_emitter_test.cc` | tests |
| add (P1) | `third_party/xnnpack/ynn_visibility.patch` + `workspace.bzl` edit | make `//ynnpack/kernels/dot` genrule outputs visible to XLA (`tf_http_archive` already supports `patch_file`; see absl/brotli workspaces for the idiom) |
| optional (P2) | ynnpack fork: `ynnpack/kernels/dot/dot.h` + `YNN_DOT_KERNEL` macro | add `const char* name` to `dot_kernel` (§7.1 shows the no-fork alternative) |

No wrapper TU is needed: the generated kernel sources (e.g.
`x86_avx512_fp32.cc`) are **self-contained** — they include only
`<immintrin.h>` and C++ std headers, define their own `YNN_ALWAYS_INLINE`, and
(for x86 f32) carry no `#if` build-predicate guard (`generator/dot_base.py
header()/generate_dot_kernels_impl`, `generator/x86.py header()`). They can be
fed to `cc_ir_header` directly.

## 3. Step A — flag
`xla.proto` (`DebugOptions` says `// Next id: 493` at the top of the message):
```proto
optional bool xla_cpu_experimental_ynn_codegen = 493;
```
Bump the `Next id:` comment to 494 (re-check both when implementing — the
number moves). `debug_options_flags.cc`: default `false`; register a
`bool_setter_for` mirroring an existing `xla_cpu_experimental_*` bool.

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
5. ISA: `TargetMachineFeatures`
   (`xla/backends/cpu/codegen/target_machine_features.h`) has **no**
   `has_avx512f()` accessor (only `has_avx512bf16/fp16`). Use the same pattern
   as `GetCppGenIrString` (`xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.cc`):
   `absl::StrContains(target.get_target_feature_string(), "+avx512f")` — and
   likewise `"+avx512bw"`, `"+avx512vl"`, `"+avx512dq"` for the full
   `arch_flag::avx512` composite the f32 kernels are registered under.

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

### 7.1 The kernel symbol name — no fork required
`get_dot_kernel` returns a fn pointer; codegen needs the **symbol name** to emit
a direct, inlinable call. The generator names every kernel deterministically
(`generator/dot_base.py generate_dot`):

```
dot_{kind}_{block_m}x{block_n}x{block_k}_{tile_m}x{tile_n}x{tile_k}_{arch}
```

in `namespace ynn` (C++-mangled, not `extern "C"`). `kind` is `fp32` for the
f32 kernels; `arch` is the generator class's arch string: `sse2`, `avx`,
`fma3`, `avx512` (the composite, not `avx512f`), `neon`. Examples that exist at
the pinned commit: `ynn::dot_fp32_5x64x1_1x16x1_avx512`,
`ynn::dot_fp32_6x16x1_1x8x1_fma3`, `ynn::dot_fp32_8x8x4_1x4x1_neon`.

The returned `struct dot_kernel` carries `block_m/block_n/block_k/tile_n/
tile_k`; `tile_m` is not a field, but every non-`transpose_a` f32 kernel has
`tile_m = 1` (all f32 tile shapes are `1x4x1`, `1x8x1`, `1x16x1`; k2/k4
variants `1x4x2`, `1x8x2`, `1x8x4`). Since the emitter also chooses which
`arch_flags` it passes to `get_dot_kernel` (one tier at a time, §7.2), it knows
`arch`, and can reconstruct the exact symbol from the struct fields alone —
**no fork needed**. The fork hook remains a nice-to-have hardening step (it
survives upstream renames of the formula) and is nearly free: the selection
optimizer already receives and tracks each candidate's name for debug logging
(`kernel_used`, `kernels/dot/dot.cc:254-295`) — the patch just copies that
string into the returned `dot_kernel`.
Resolve the mangled name via `llvm::Module::getFunction` over the linked
bitcode module by substring match on the unmangled name (the bitcode has
exactly one definition per kernel), or itanium-mangle `ynn::<name>` directly.

### 7.2 Select the kernel at compile time
In the emitter, link host ynnpack and call (real signature, Appendix). Enum
spelling from `ynnpack/include/ynnpack.h`: `ynn_type_fp32` (also
`ynn_type_bf16`, `ynn_type_int8`, … — plain C enum, no scope).
```cpp
ynn::dot_type type{ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};
ynn::dot_shape shape; shape.m = M; shape.n = N; shape.k1 = K;
// Pin tile_k=1 so the packed layout below has no K interleave and no K tail
// (all f32 x86/NEON kernels have tile_k==1; the avx2/avx512 _k2/_k4 variants
// exist but requesting tile_k=1 excludes them — the optimizer rejects
// tile_k != required_tile_k, kernels/dot/dot.cc:275). block_n=0 = "any";
// ynnpack itself uses dot_packed_shape{0, 1} this way (`no_tile_k`,
// subgraph/dot.cc:929).
ynn::dot_packed_shape packed_shape{/*block_n=*/0, /*tile_k=*/1};
uint64_t arch = MapTargetToYnnArchFlags(target);   // exactly one tier, §5.2 of the design doc
ynn::dot_kernel k = ynn::get_dot_kernel(type, shape, &packed_shape,
    /*required_flags=*/0, /*transpose_a=*/std::make_optional(false), arch);
```
`MapTargetToYnnArchFlags` — one tier, matching how ynnpack compiles the
kernels (`ynnpack/build_defs.bzl` `_YNN_PARAMS_FOR_ARCH`):
- feature string has `+avx512f,+avx512bw,+avx512vl,+avx512dq` →
  `ynn::arch_flag::avx512` (composite `= avx512f|avx512bw|avx512vl|avx512dq`)
  plus its implied lower tiers;
- else has `+fma` and `+avx` → `ynn::arch_flag::fma3 | avx | sse2…` (ynnpack's
  f32 middle tier is **fma3**, compiled `-mavx -mfma -mno-avx2`; runtime
  detection in `base/arch.cc` is just `cpuinfo_has_x86_fma3()`);
- aarch64 → `ynn::arch_flag::neon` (baseline arm64, no extra copts).

### 7.3 Pack constant B at compile time
B is `(K,N)` f32. The packed layout, from the pack node in
`ynnpack/subgraph/dot.cc` (`make_pack_impl`, ~line 372) and `kernels/dot/pack.h`:

```
packed(ki, ni, ko, no) = B(ko*tile_k + ki, no*block_n + ni), zero-padded
dims innermost→outermost (all contiguous):
  ki: extent tile_k              stride 4 (f32)
  ni: extent block_n             stride tile_k*4
  ko: extent ceil(K/tile_k)      stride block_n*tile_k*4
  no: extent ceil(N/block_n)     stride roundup(K,tile_k)*block_n*4
total size = roundup(K, tile_k) * roundup(N, block_n) floats
```

The subgraph constructs the packer as `packer(transpose, elem_bits,
/*tile_m=*/tile_k, /*tile_n=*/block_n)` and calls `pack(k, n, input_stride,
input, /*output_stride=*/ko_stride, /*output_block_stride=*/no_stride, output)`
(`subgraph/dot.cc:417-422`). For our slice:
```cpp
int64_t Ko = CeilDiv(K, k.tile_k), No = CeilDiv(N, k.block_n);
std::vector<float> packed(Ko * k.tile_k * No * k.block_n);   // zero-init
ynn::packer pk(/*transpose=*/false, /*elem_size_bits=*/32,
               /*tile_m=*/k.tile_k, /*tile_n=*/k.block_n);
pk.pack(/*m=*/K, /*n=*/N, /*input_stride=*/N * 4, b_data,
        /*output_stride=*/k.block_n * k.tile_k * 4,          // ko stride
        /*output_block_stride=*/Ko * k.tile_k * k.block_n * 4,  // no stride
        packed.data());
```
With `tile_k == 1` (§7.2) this degenerates to: each `no` block is a `K x
block_n` row-major slab of B's columns `[no*block_n, (no+1)*block_n)`,
zero-padded in N.

Bake `packed` as a private `llvm::GlobalVariable` (ConstantDataArray, align 64);
keep its `ptr`. (Non-constant B → fall back in Phase 1.)

### 7.4 Extract the dot kernel as bitcode
No wrapper TU: compile the generated kernel source **directly**. The
`ynn_generate_srcs` genrule (`@XNNPACK//ynnpack/kernels/dot:x86_fp32`) outputs
`x86_avx512_fp32.cc` / `x86_fma3_fp32.cc` (plus `.inc`s) as ordinary file
targets in that package; the `.cc` files are self-contained (§2) — no include
paths, no `YNN_ARCH_*` defines needed (those macros only gate `kernels.inc`,
which we don't compile).
```python
load("//xla/codegen/intrinsic/cpp:cc_to_llvm_ir.bzl", "cc_ir_header")
cc_ir_header(
    name = "ynn_dot_f32_avx512_ll",
    src = "@XNNPACK//ynnpack/kernels/dot:x86_avx512_fp32.cc",
    namespace = "ynn_kernels",
    base_name = "ynn_dot_f32_avx512",
    # Mirror ynnpack: ynn_kernel_copts(unroll_loops=False) + x86_avx512 arch
    # copts (build_defs.bzl). cc_ir_header's own base flags are
    # "-emit-llvm -O3 -DNDEBUG -mprefer-vector-width=512 -std=c++17 ..." with
    # user copts appended, so the trailing -O2 takes effect. The JIT
    # re-optimizes after inlining, so -fno-unroll-loops keeps the bitcode small
    # and lets LLVM unroll with the real constant trip counts instead.
    copts = ["-O2", "-fno-unroll-loops",
             "-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq"],
)
# likewise: ynn_dot_f32_fma3_ll  <- x86_fma3_fp32.cc, copts -mavx -mfma -mno-avx2
#           ynn_dot_f32_neon_ll  <- arm64_neon_fp32.cc, no extra -m copts
```
Generates `inline const std::string kYnnDotF32Avx512Ir` in `ynn_kernels`.
**Visibility:** `ynnpack/kernels/dot/BUILD` has
`package(default_visibility = ["//ynnpack:__subpackages__"])`, so the genrule
outputs are not visible to XLA out of the box. Add
`third_party/xnnpack/ynn_visibility.patch` making that package (or just the
`x86_fp32`/`arm_fp32` genrules) public, and pass it via `patch_file` in
`third_party/xnnpack/workspace.bzl` (`tf_http_archive` supports this; see
`third_party/absl/workspace.bzl` for the idiom). Get one target emitting
non-empty bitcode in isolation first
(`bazel build :ynn_dot_f32_avx512_ll && llvm-dis` the `.llvmbc` section).

### 7.5 Emit the driver and inline the kernel
Replace `EmitNaiveGemm` with `EmitYnnDotDriver`:
1. Declare the kernel with the exact ABI (Appendix `dot_kernel_fn`):
   `void(i64×8, ptr, i64×3, ptr, i64, ptr, i64, ptr)` — i.e. `(m,n,k3,k2,k1,
   a_stride_m,a_stride_k3,a_stride_k2, a, b_stride_k3,b_stride_k2,b_stride_k1, b,
   c_in_stride_m, c_in, c_out_stride_m, c_out)`. Use `module->getOrInsertFunction(
   sym, fnty)` where `sym` is the mangled symbol reconstructed per §7.1.
2. Pointers from the prototype: `a` = `proto.arguments[0]`'s
   `IrArray::GetBasePointer()` (`xla/service/llvm_ir/ir_array.h`); `c_out` =
   `proto.results[0]` likewise; `b` = packed-B global (§7.3); `c_in` per step 4.
3. Emit a **simple M×N-blocked loop nest** (correctness first; `schedule_dot`
   cache tiling is a Phase-2 perf add). Partition M blocks over
   `proto.workgroup_id.x`; inside, loop `mb` step `k.block_m` and `no` step
   `k.block_n`, calling the kernel once per `(mb, no)` with
   `m = min(block_m, M-mb)`, `n = min(block_n, N-no*block_n)`.
   **Why the driver must loop N too:** each packed `no` block is a separate
   `roundup(K,tile_k) × block_n` slab; inside one call the kernel's own N walk
   advances `b` by `block_n*tile_k` elements (the *unpacked* layout), so a
   call must never span two packed blocks. This is exactly what
   `block_dot_n` does (`schedule.h:64-73`): `n = min(n, block_n)` per call,
   then `b += b_stride_n * block_n` (= one slab). Per-call pointers:
   `b = packed_global + no*no_stride` (§7.3), `c_out`/`c_in` offset by
   `no*block_n*4` columns (bias too — its stride-0 trick broadcasts rows, not
   columns). The masked-tail N loads inside the kernel stay in-bounds because
   packing zero-pads N to `block_n`.
   K-split, resolved from `make_dot_impl` (`subgraph/dot.cc:84-134`): a 2-D dot
   has `k2=k3=1` with `a_stride_k2=a_stride_k3=b_stride_k2=b_stride_k3=0`, and
   `k1` must be a multiple of `tile_k` (the subgraph handles `K % tile_k` with
   a zero-padded A `alloca`). **With `tile_k` pinned to 1 (§7.2), `k1 = K`
   exactly and the tail path does not exist for this slice.** Remaining
   strides: `a_stride_m = K*4`; `b_stride_k1 = ko_stride/tile_k = block_n*4`
   (`subgraph/dot.cc:129` divides by `tile_k`); `c_in_stride_m` per step 4;
   `c_out_stride_m = N*4`. `NumWorkGroups.x = ceil_div(M, block_m * <chunk>)`.
4. Bias / `c_in` — resolved from the generated kernel bodies
   (`generator/dot_base.py`): accumulators are **zero-initialized in
   registers** (`init_c_tile` = `setzero`) and `C_in` is added at store time
   inside an `if (C_in) { ... }` guard (`add_c_block`); pointer advancement is
   also null-guarded. So:
   - no bias → pass `c_in = null`, `c_in_stride_m = 0` — natively supported;
   - bias `[N]` → pass the bias buffer as `c_in` with **`c_in_stride_m = 0`**
     (row-broadcast: `c_in_ptr(i,j) = C_in + i*stride_m + j*4`, so stride 0
     reads the same N-row for every i) — bias fusion costs zero extra IR;
   - full addend `(M,N)` → `c_in` = addend base, `c_in_stride_m = N*4`.
   The only unsupported broadcast is along **n** (scalar-per-row); the
   subgraph copies that into `c_out` first (`subgraph/dot.cc:203-215`) — out
   of slice, fall back.
5. Link + inline:
   ```cpp
   xla::codegen::CppGenIntrinsicLibrary lib(ynn_kernels::kYnnDotF32Avx512Ir,
                                            "ynn_dot");
   lib.LinkIntoModule(*module);
   // sym = the C++-mangled name of ynn::dot_fp32_{...} reconstructed per §7.1
   // (or find it by unmangled-substring scan over the linked module's
   // definitions before LinkIntoModule internalizes them).
   xla::codegen::GetCppGenFunction(module, sym);  // InternalLinkage+AlwaysInline
   ```
   `IrCompiler` inlines the kernel, constant-folds dims, emits the object.

**Checkpoint (Phase 1):** the JIT'd kernel inlines the ynnpack dot kernel;
numerics match; FileCheck sees `call @<sym>` (mangled
`ynn::dot_fp32_..._avx512`) pre-inline.

### 7.6 Phase 2 (optimization, later)
- Activation epilogue: emit the elementwise neighbors on the `c_out` tile (ynnpack
  `kernels/elementwise` inlined, or `CpuElementalIrEmitter`) before store; LLVM
  fuses into the kernel epilogue.
- Use `ynn::schedule_dot` + `ynn::run_dot` (header-only) for cache-aware tiling.
- bf16 (`x86_avx512bf16_*`), the `fma3` tier (AVX+FMA3, `-mno-avx2`) and NEON
  (runtime-selected like `GetCppGenIrString`), `dot_flag::transpose_a` kernels
  (with A transpose), `_k2`/`_k4` K-interleaved kernels.

## 8. BUILD deps
`ynn_kernel_emitter` deps += `@XNNPACK//ynnpack/kernels/dot:dot` (host
`get_dot_kernel`), `@XNNPACK//ynnpack/kernels/dot:pack` (host `packer`),
`:ynn_dot_f32_avx512_ll`, `//xla/codegen/intrinsic/cpp:_cpp_gen_intrinsics`.
Visibility: XLA's existing deps use only the top-level `@XNNPACK//ynnpack`
and `@XNNPACK//ynnpack:ynnpack_h`, which are explicitly
`//visibility:public`. The internal `//ynnpack/kernels/dot:{dot,pack}` targets
and the genrule outputs sit under
`package(default_visibility = ["//ynnpack:__subpackages__"])`, so the
`ynn_visibility.patch` from §7.4 must cover all three.

## 9. Tests (`ynn_kernel_emitter_test.cc`)
1. **Numerics:** one `f32` constant-RHS matmul; flag on vs off; aligned and
   unaligned `M,N` (M tail vs `block_m`, N tail vs `block_n`/`tile_n`; K has no
   tail with `tile_k=1`). Add a biased case (bias `[N]`, `c_in_stride_m=0`) and
   a no-bias case (`c_in=null`).
2. **FileCheck/IR** (mirror `tanh_test`/`eigen_unary_test`): pre-opt module has
   `call @<sym>` + the packed-B global; post-opt it's inlined.
3. **Fallback:** bf16 / non-constant RHS / transpose dot with flag on still runs
   via `YnnFusionThunk`, matches reference.
4. **Benchmark:** extend `benchmarks/ynn_fusion_benchmark_test.cc`.

## 10. Known simplifications → later
| Shortcut | Why OK | Later |
|---|---|---|
| packed B baked as LLVM global | correct, simple | constant buffer via HLO rewrite; drop unused RHS arg |
| simple M×N-block loop | correct | `schedule_dot`/`run_dot` cache tiling |
| `tile_k` pinned to 1 | no K tail; simplest packed layout | allow `_k2`/`_k4` kernels (K-interleaved pack, padded-A tail) |
| RHS constant only | enables compile-time pack | prologue pack node for dynamic B |
| no activation | bias via `c_in` (stride 0) only | P2 epilogue fusion |
| one kernel/tier hardcoded | proves path | fma3 tier + runtime select (`GetCppGenIrString` pattern) |
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
`block_dot_{m,n,k}(...)` (all three null-guard `init_c` when advancing it).
`ynnpack/base/arch.h`: `arch_flag::{fma3=1<<6, avx512f=1<<7, avx512bw,
avx512vl, avx512dq, avx512bf16, avx512vnni, amxbf16, amxint8, sme, sme2, …}`,
composite `avx512 = avx512f|avx512bw|avx512vl|avx512dq`,
`uint64_t get_supported_arch_flags()`.

Reference integration: `ynnpack/subgraph/dot.cc` (`make_dot_impl`) — sets
`packed_shape{block_n, tile_k}`, calls `get_dot_kernel`, wraps `kernel.kernel` in
a lambda, drives via `schedule_dot`+`run_dot`, bias via `c_in`, handles the
`k % tile_k` tail with a padded-A `alloca`.

### Resolved facts (read from the pinned source)

**f32 kernel catalog** (`generator/x86_fp32.py`, `generator/arm_fp32.py`;
symbols per §7.1's formula, all `namespace ynn`, all `tile_m=1`):

| arch (class string) | arch_flag required | tile (m,n,k) | block shapes (m,n,k) |
|---|---|---|---|
| `avx512` | composite `avx512` | 1,16,1 | 1–5×64×1, 1–5×32×1, 5×16×1 |
| `fma3` | `fma3` | 1,8,1 | 1–2×32×1, 1–6×16×1, 8×8×1 |
| `avx` | `avx` | 1,8,1 | 1–2×32×1, 1–4×16×1, 4/6/8×8×1 |
| `sse2` | `sse2` | 1,4,1 | 1–3×16×1, 1–4×8×1, 4/6/8×4×1 |
| `neon` (arm64) | `neon` | 1,4,1 | 1–3×32×4, 1–5×16×4, 1/4/6/8×8×4, 8×4×4 |

Plus packed-only variants with K-interleave: `x86_avx2_fp32_k2`/
`x86_avx2_fma3_fp32_k2` (tile 1,4,2), `x86_avx512_fp32_k2` (tile 1,8,2),
`x86_avx512_fp32_k4` — excluded from the Phase-1 slice by requesting
`dot_packed_shape{0, /*tile_k=*/1}`. `fma3`/`avx512` f32 kernels carry
`dot_flag::consistent_arithmetic` (pure FMA); `avx`/`fma3`/`avx512` carry
`dot_flag::unaligned_b` (masked loads tolerate unpadded B).

**Per-arch compile flags** (`ynnpack/build_defs.bzl` `_YNN_PARAMS_FOR_ARCH` +
`ynn_kernel_copts(unroll_loops=False)` = `-O2 -fno-unroll-loops`):

| arch group | copts | define (auto: `YNN_ARCH_`+upper, + implied) |
|---|---|---|
| `x86_avx512` | `-mavx512f -mavx512bw -mavx512vl -mavx512dq` | `YNN_ARCH_X86_AVX512` (+FMA3/AVX2/F16C…) |
| `x86_fma3` | `-mavx -mfma -mno-avx2` | `YNN_ARCH_X86_FMA3` (+AVX…) |
| `arm64_neon` | *(none — baseline arm64)* | `YNN_ARCH_ARM64_NEON` |

The defines only gate `kernels.inc` / `dot.cc`; the generated kernel `.cc`
files themselves have no guards and no ynnpack includes (only `<immintrin.h>`
or `<arm_neon.h>` + std) — compile them standalone with just the copts.

**Generated-kernel semantics** (`generator/dot_base.py`): accumulators
zero-init in registers; K1→K2→K3 loops accumulate; `finalize_c_block`; then
`if (C_in) add`, store. Null `C_in` fully supported. `c_in_ptr/c_out_ptr` use
`min(i, M-1)` row clamping (safe partial-M). A is addressed
`A + i*a_stride_m + k1*4` (row-major, k contiguous); the kernel asserts
`M <= block_m` and internally loops N (`while (N >= block_n)` then a masked
`tile_n` tail) — but its internal N advance (`b += block_n*tile_k` elements)
assumes the unpacked layout, so with packed B every call must have
`n <= block_n` (§7.5 step 3).

**Packed-B binding to the kernel ABI** (`subgraph/dot.cc:84-139`):
`tile_k = b_ki.extent()`; `b_stride_k1 = ko_stride / tile_k` (= `block_n*4`
when contiguous); `b_stride_n = no_stride / block_n` (= `roundup(K,tile_k)*4`)
is a "lie" stride valid only in `block_n` steps, used by `run_dot`'s
n-blocking, never inside a kernel call; `k1` rounded down to `tile_k`, tail
via padded-A copy (absent when `tile_k==1`).

**Missing-bias binding** (`subgraph/dot.cc:1050-1056`,
`subgraph/runtime.cc:393-400`): a node without an initializer binds
`runtime.null_buffer()` — a rank-0 constant `raw_buffer` with null base — and
that null flows through `run_dot` into the kernels' `if (C_in)` guard.

**XLA-side anchors**: `DebugOptions` next id 493 (`xla/xla.proto:1614`);
`TargetMachineFeatures::get_target_feature_string()`
(`xla/backends/cpu/codegen/target_machine_features.h:72`);
`IrArray::GetBasePointer()` (`xla/service/llvm_ir/ir_array.h:261`);
`tf_http_archive(..., patch_file=...)` (`third_party/repo.bzl:61`) with
`@XNNPACK` pinned in `third_party/xnnpack/workspace.bzl`.
