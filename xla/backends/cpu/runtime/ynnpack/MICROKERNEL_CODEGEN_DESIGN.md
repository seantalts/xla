# Design: LLVM-IR microkernel codegen for the XLA:CPU ynnpack path

Status: draft / RFC (v6 — grounded in the ynnpack source at the pinned commit)
Scope: XLA:CPU backend, `__ynn_fusion` path

> Terminology: **ynnpack** is a from-scratch successor to XNNPACK (a "spiritual
> successor", per its README), in the `ynnpack/` tree of the `google/XNNPACK`
> repo. It has **its own** kernels in `ynnpack/kernels/` (generated C++ with SIMD
> intrinsics), a subgraph API (`ynnpack/include/ynnpack.h`), and uses **slinky**
> for cross-node loop fusion. It does **not** reuse classic XNNPACK microkernels
> (`src/…`, `xnn_*_ukernel`), and neither does this design.

## 1. Goal

Today XLA:CPU offloads matched fusions to ynnpack and lets it *execute* them
(subgraph → slinky loops → ynnpack kernels). This proposal changes the back
half: **emit the ynnpack kernels as LLVM IR** into a fused driver kernel that XLA
JIT-compiles, run via a plain `KernelThunk`. The extract→embed→link→inline
machinery already exists in-tree (`xla/codegen/intrinsic/cpp/`, used for `tanh` /
Eigen unary); ynnpack kernels are generated intrinsic-C++, so they lift to
bitcode through it.

The win is what the offload model hides from the compiler once the kernel is IR:
- **Epilogue fusion** — ynnpack's dot kernel has no fused activation (it computes
  `C_out = C_in + A·B`); the bias is just `C_in`, and activations are *separate*
  elementwise nodes today. With the kernel inlined, fold those onto the output
  tile **in registers** before the store.
- **Shape/target specialization** — compile-time loop bounds; K unrolls; exact
  `-mcpu`/features (`TargetMachineFeatures`).
- **One runtime/threadpool** — XLA workgroup grid (`KernelThunk` → Eigen pool),
  not slinky via `SlinkyThreadPool`.
- **AOT + object cache**; **autotuning** of driver tiles.

## 2. How the path works today

```
HLO → LibraryRewriter+YnnMatcher → HloFusion(kCustom, "__ynn_fusion")
    → ThunkEmitter::EmitYnnFusionThunk → YnnFusionThunk
    → EmitYnnSubgraph (ynn_define_*) → ynn_optimize_subgraph
    → ynn_create_runtime → ynn_reshape_runtime → ynn_invoke_runtime
       (slinky fuses/schedules loops, calls ynnpack kernels, SlinkyThreadPool)
```

Facts that shape the design:
1. **ynnpack doesn't emit code; slinky is an interpreter.** Per the README,
   ynnpack = subgraph API + kernels (no operator API); slinky runs the loops
   "outside the microkernels"; "packing weights is handled by a subgraph node
   (that may or may not get constant folded)."
2. **The public API is opaque** (§3).

## 3. The ynnpack layers we depend on (read from source)

Public API (`ynnpack/include/ynnpack.h`): ~30 `ynn_define_*` ops;
`ynn_optimize_subgraph` (status only, **no plan introspection**);
`ynn_create_runtime/reshape/invoke`; `ynn_query_runtime` (only `concurrency`).
No packing or codegen in the public API. So we reach to the **kernel layer**:

**Dot kernel ABI** — `ynnpack/kernels/dot/dot.h`:
```cpp
// C_out(i,j) = C_in(i,j) + sum_{k3,k2,k1} A(i,k3,k2,k1) * B(k3,k2,k1,j)
typedef void (*dot_kernel_fn)(size_t m, size_t n, size_t k3, size_t k2, size_t k1,
    size_t a_stride_m, size_t a_stride_k3, size_t a_stride_k2, const void* a,
    size_t b_stride_k3, size_t b_stride_k2, size_t b_stride_k1, const void* b,
    size_t c_in_stride_m, const void* c_in,
    size_t c_out_stride_m, void* c_out);
```
K is split 3 ways (k3,k2,k1); `m ≤ block_m` (caller loops); B is **packed**;
`c_in` is the accumulator/bias; **no activation/clamp in the kernel**. Some
kernels set `dot_flag::transpose_a` (e.g. SME) requiring a transposed A layout.

**Selection** — `ynn::get_dot_kernel(const dot_type&, const dot_shape&,
const dot_packed_shape*, uint32_t required_flags, optional<bool> transpose_a,
uint64_t arch_flags)` returns:
```cpp
struct dot_kernel { dot_kernel_fn kernel; int block_m, block_n, block_k,
                    tile_n, tile_k; uint32_t flags; float cost; };
```
`arch_flags` (`ynnpack/base/arch.h`: `arch_flag::{avx512f,fma3,avx2,amxbf16,
amxint8,sme,sme2,…}`) constrains the choice to a target ISA — pass the XLA
target's flags, not the host's.

**Scheduling** — `ynnpack/kernels/dot/schedule.h` (header-only templates):
`schedule_dot(cache_sizes, m, n, ks, block_m, block_n, block_k, …)` returns
`dot_loop{dim∈{m,n,k}, blocks}`; `run_dot(...)` / `block_dot_{m,n,k}` drive the
cache-blocked loop nest and chain `c_in→c_out` for split-K. This is ynnpack's
*own* dot tiling (slinky only fuses across nodes).

**Packing** — `ynnpack/kernels/dot/pack.{h,cc}`: `class packer(bool transpose,
size_t elem_size_bits, size_t tile_m, size_t tile_n)` + `pack(m, n, in_stride,
in, out_stride, out_block_stride, out)`. `packed(mi,ni,mo,no) =
input(mo*tile_m+mi, no*tile_n+ni)`, padded to tiles.

**Kernels are generated & per-arch** — `ynnpack/kernels/dot/kernels.inc`
`#include`s generated `.inc` files (e.g. `x86_avx512_fp32.inc`, `x86_fma3_fp32.inc`)
produced by `ynnpack/kernels/dot/generator/*.py` via `ynn_generate_srcs`, and
compiled into `dot.cc` per-arch with `ynn_kernel_copts`. So one ISA tier =
`dot.cc` built with one arch define (`YNN_ARCH_X86_AVX512`) + `-mavx512f`.

## 4. Target architecture

```
HLO fusion (__ynn_fusion)                         ← unchanged front half
  ▼
YnnKernelEmitter : KernelEmitter<LlvmKernelSource>    ← NEW
  - EmitKernelPrototype() via KernelApiIrBuilder → XLA_CPU kernel ABI
  - select kernel: ynn::get_dot_kernel(type, shape, arch_flags) → block/tile/sym
  - pack constant B at compile time (ynn::packer) → baked constant
  - emit the schedule.h loop nest (or call run_dot), partitioned by workgroup_id;
    per block call the dot kernel (bias via c_in), then the fused elementwise
    epilogue in registers
  - link the dot-kernel bitcode (CppGenIntrinsicLibrary), inline + specialize
  - return KernelDefinition{ KernelSpec, LlvmKernelSource }
  ▼
JitCompiler (IrCompiler: opt/target/fast-math) → KernelThunk (Eigen workgroups)
```
Everything from `JitCompiler` down already exists (`DotKernelEmitter` /
`ElementalKernelEmitter` use it). The genuinely new work is **reproducing the
loop nest slinky+schedule.h produce at runtime** — trivial for one dot
(`schedule.h` is reusable header-only code), harder for fused subgraphs (the impl
slice starts with one dot, falls back otherwise).

## 5. The core plan

### 5.1 Reuse the in-tree extract→embed→link machinery
`cc_ir_header` (`xla/codegen/intrinsic/cpp/cc_to_llvm_ir.bzl`) compiles a TU with
`-emit-llvm -O3 …` and embeds the `.llvmbc` as `inline const std::string k<Name>Ir`;
`CppGenIntrinsicLibrary::LinkIntoModule` links it into the kernel module with
`InternalLinkage + AlwaysInline`; `IrCompiler` inlines + specializes. We add a
`cc_ir_header` target that builds the ynnpack `dot.cc` for one ISA tier (§3,
generated sources + `ynn_kernel_copts` + arch define).

### 5.2 Variant count: bounded ISA tiers, runtime-selected
ynnpack already compiles per-arch; we mirror the Eigen precedent (ship a few
tiers, select at runtime by `TargetMachineFeatures`, as `GetCppGenIrString`
does). The compiled kernel keeps only the selected symbol (linker pulls only
referenced functions). For AOT, embed only the target tier.

### 5.3 Selection: a ynnpack fork hook (decided, now concrete)
`get_dot_kernel` returns a function *pointer*, but codegen needs the kernel's
**symbol name** to declare + link. Minimal fork patch: add `const char* name;`
to `struct dot_kernel` (populated from the `YNN_DOT_KERNEL` macro / generator)
so XLA can call `ynn::get_dot_kernel(...)` at compile time and learn the symbol
+ `block_*`/`tile_*` to emit against. (Until then, a slice can hardcode one
kernel symbol — §5.5.)

### 5.4 Reuse vs re-emit the schedule
Two ways to get the loop nest:
- **(reuse)** compile a thin driver TU that calls `ynn::schedule_dot` +
  `ynn::run_dot` with the selected kernel, and extract *that* (driver + kernel)
  as one bitcode blob via `cc_ir_header`; XLA wraps it with the kernel ABI
  prototype + workgroup partition + epilogue. Maximizes reuse of ynnpack's
  tiling.
- **(re-emit)** emit the `block_dot_{m,n,k}` blocking directly in LLVM IR for
  finer control / fusion. More work.
Recommend **reuse** first.

### 5.5 Option A (XLA emits its own microkernel) — fallback only
Extend `tiled_dot_emitter`/`VectorIrBuilder`; kept only where a ynnpack kernel
can't be extracted. Doesn't deliver ynnpack's tuning.

### 5.6 Honest gaps (now small)
- **Reproducing slinky's cross-node schedule** for fused subgraphs (one dot is
  fine via `schedule.h`).
- **A-transpose kernels** (`dot_flag::transpose_a`, e.g. SME) need a transposed A
  layout — restrict the slice to non-transpose kernels (x86 f32 AVX-512).
- **Generated-source build wiring** into `cc_ir_header` (the main build task).
- *(was a worry, now minor)* **asm** — f32 x86 dot kernels are generated AVX-512
  intrinsics (liftable); only exotic paths (some AMX/SME) might not be, and those
  use the extern-call fallback.

## 6. LLVM version: no pinning, by construction
`cc_ir_header` compiles with the build's clang and embeds the result;
`LinkIntoModule` links into the JIT module built by the same LLVM; bitcode is
regenerated every build (never checked in). Nothing to pin — the tanh/Eigen
precedent works this way today.

## 7. Supporting design points

### 7.1 Epilogue / bias fusion (clean here)
- **Bias** is `c_in` to the dot kernel (`C_out = C_in + A·B`) — pass the bias
  buffer (or zero) as `c_in`; no separate pass.
- **Activation/clamp/other elementwise** are separate nodes today; with the dot
  kernel inlined, emit them on the `c_out` tile (ynnpack `kernels/elementwise`
  inlined the same way, or XLA `CpuElementalIrEmitter`) before the store. LLVM
  fuses them in.

### 7.2 Packing (ynn::packer at compile time)
For constant B, call `ynn::packer(transpose=false, elem_bits=32, tile_m, tile_n)`
+ `pack(...)` at emit time (ynnpack is linked into XLA) with the kernel's
`tile_n`/`tile_k` (and `dot_packed_shape`), materialize the packed bytes as a
constant the kernel reads. Non-constant B → prologue pack or fall back.

### 7.3 Selection, gating, fallback
Reuse `YnnMatcher`/`ynn_support` for *what* fuses. Add a default-off flag
(`xla_cpu_experimental_ynn_codegen`). Emitter returns `absl::Status`; on anything
unsupported **fall back to `YnnFusionThunk`**.

### 7.4 Dynamic shapes / 7.5 Threading
Static shapes first (else runtime bounds or fall back). Drop `SlinkyThreadPool`
from the hot path; parallelism = `KernelThunk` workgroups over output tiles
(mirrors `schedule.h`'s outer loop); keep slinky only for fallback fusions.

## 8. Phasing

1. **Seam.** `YnnKernelEmitter` end-to-end on a naive XLA-emitted `f32` matmul
   (no ynnpack kernel), default-off flag, route in `ThunkEmitter`, else fall
   back. Validates emitter→JIT→`KernelThunk`.
2. **ynnpack dot kernel.** `cc_ir_header` for x86 AVX-512 `f32` `dot.cc`; select
   via `get_dot_kernel`; pack constant B with `ynn::packer`; drive via `run_dot`
   (reuse) or emitted blocking; inline.
3. **Epilogue fusion.** bias via `c_in`; fold elementwise neighbors; add bf16.
4. **Fork hook + tiers.** add `dot_kernel.name`; AVX2/FMA3 tiers; runtime select.
5. **Coverage.** reduce/conv via the corresponding `ynnpack/kernels/`; SME (with
   A-transpose); autotuning.

## 9. Risks / open questions
- **Reproducing slinky's schedule** for fused subgraphs (bounds near-term scope).
- **Generated-source → cc_ir_header** build wiring (main build task).
- **Fork hook** maintenance (tiny: expose the symbol name).
- **A-transpose kernels** (avoid in the slice).
- *(Resolved)* LLVM pinning; *(minor for ynnpack)* asm.

## 10. Decisions needed
1. **Schedule strategy (§5.4):** reuse `run_dot` (recommended) vs re-emit
   blocking in IR.
2. **First tier + op:** x86 AVX-512 `f32` dot (recommended).
3. **Scope:** design only vs land the Phase-1/2 slice.

*Decided:* runtime = `KernelThunk`; selection = ynnpack fork hook exposing the
kernel symbol (§5.3); kernel source = ynnpack's own `ynnpack/kernels/`, not
classic XNNPACK.
