# XLA:CPU Runtime Surface Area

This document describes the runtime surface area of the XLA CPU backend: what
gets executed, what libraries are linked, and how the pieces fit together.

## Architecture Overview

XLA:CPU compiles HLO programs through two paths that converge at runtime:

1. **Thunk-based execution** -- The HLO graph is lowered to a sequence of
   `Thunk` objects (one per operation or fused subgraph). A `ThunkExecutor`
   runs them with DAG-based scheduling, enabling concurrent execution of
   independent thunks.

2. **JIT-compiled kernels** -- Element-wise and fused operations are compiled
   to native code via LLVM. The compiled code calls back into runtime support
   functions (prefixed `__xla_cpu_runtime_`) for operations that are too
   complex to inline.

The entrypoint for users is the **PJRT CPU client**, either linked directly or
loaded as a shared library plugin.

```
User code
  -> PJRT C API  (GetPjrtApi)
    -> PjRtCpuClient
      -> CpuCompiler  (HLO -> ThunkSequence + LLVM JIT kernels)
        -> ThunkExecutor (runs thunks, resolves buffers, manages concurrency)
          -> Individual Thunks (dot, conv, collective, custom call, kernel, ...)
            -> Eigen / oneDNN / XNNPACK / LLVM-compiled kernels
```

## Key Source Locations

| Component | Path |
|---|---|
| PJRT CPU client | `xla/pjrt/cpu/cpu_client.h` |
| PJRT plugin entry | `xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h` |
| PJRT C API + `.so` | `xla/pjrt/c/pjrt_c_api_cpu.cc`, `pjrt_c_api_cpu_internal.cc` |
| CPU compiler | `xla/service/cpu/cpu_compiler.h` |
| CPU executable | `xla/service/cpu/cpu_executable.h` |
| Thunk base class | `xla/backends/cpu/runtime/thunk.h` |
| Thunk executor | `xla/backends/cpu/runtime/thunk_executor.h` |
| Thunk emitter | `xla/service/cpu/thunk_emitter.h` |
| Runtime symbols | `xla/service/cpu/cpu_runtime.h` |
| Function library | `xla/backends/cpu/runtime/function_library.h` |
| CPU FFI bindings | `xla/backends/cpu/ffi.h` |
| Collectives | `xla/backends/cpu/collectives/` |
| oneDNN thunks | `xla/backends/cpu/runtime/onednn/` |
| XNNPACK thunks | `xla/backends/cpu/runtime/ynnpack/` |

## PJRT Plugin Loading

The CPU backend ships as a shared library (`pjrt_c_api_cpu_plugin.so`). The
version script (`pjrt_c_api_cpu_version_script.lds`) exports exactly **one
symbol**:

```
VERS_1.0 {
  global:
    extern "C" { GetPjrtApi; };
  local: *;
};
```

Frameworks load the plugin with:

```cpp
void* library = dlopen("pjrt_c_api_cpu_plugin.so", RTLD_LAZY);
auto init_fn = (const PJRT_Api* (*)()) dlsym(library, "GetPjrtApi");
const PJRT_Api* api = init_fn();
```

The returned `PJRT_Api*` struct contains function pointers for the full PJRT
interface (compile, execute, buffer management, etc.) plus extension chains
for FFI, layouts, shardings, memory descriptions, and phased compilation.

Alternatively the CPU client can be linked directly via
`GetXlaPjrtCpuClient(CpuClientOptions)`.

## Thunk Types (Runtime Operations)

Every HLO operation maps to a `Thunk` subclass. These are the CPU thunk kinds
defined in `xla/backends/cpu/runtime/thunk.h`:

| Thunk Kind | Purpose |
|---|---|
| `kKernel` / `SmallKernel` | Execute JIT-compiled host kernels (fusions, element-wise ops) |
| `kDot` | Matrix multiplication (dispatches to Eigen or oneDNN) |
| `kConvolution` | Convolution (dispatches to Eigen, oneDNN, or ACL) |
| `kFft` | Fast Fourier Transform (via DUCC) |
| `kSort` | Sorting with compiled comparator |
| `kTopK` | Top-K selection |
| `kCopy` | Buffer-to-buffer copy |
| `kCall` | Call into a sub-computation's thunk sequence |
| `kConditional` | Branch dispatch |
| `kWhile` | Loop execution |
| `kCustomCall` | User-registered custom operations (untyped or FFI) |
| `kCollective` | All-reduce, all-gather, reduce-scatter, all-to-all, collective-permute |
| `kInfeed` / `kOutfeed` | Host-device data transfer queues |
| `kReplicaId` / `kPartitionId` | SPMD identity queries |
| `kRngGetAndUpdateState` | RNG state management |
| `kOneDnnFusion` | Fused operations via oneDNN (matmul+bias+activation, etc.) |
| `kYnnFusion` | Fused operations via XNNPACK/YNNPACK |

The `ThunkExecutor` analyzes `BufferUse` and `ResourceUse` declarations from
each thunk to build a DAG and execute independent thunks concurrently via a
configurable `TaskRunner`.

## JIT Runtime Symbol Table

When XLA compiles HLO to native code via LLVM, the generated code calls
runtime support functions by name. All symbols use the prefix
`__xla_cpu_runtime_` (defined in `xla/service/cpu/cpu_runtime.h`).

These are resolved in two ways:
1. **JIT mode**: `RuntimeSymbolGenerator` maps names to function pointers at
   JIT compile time.
2. **AOT mode**: The linker resolves them from the `cpu_runtime` library.

### Complete Symbol List

**Matrix multiplication (Eigen)**
- `__xla_cpu_runtime_EigenMatMul{F16,F32,F64,C64,C128,S32,U8}`
- `__xla_cpu_runtime_EigenBatchMatMulF32`
- `__xla_cpu_runtime_EigenSingleThreadedMatMul{F16,F32,F64,F8E4M3FN,F8E5M2,C64,C128,S32,U8}`

**Matrix multiplication (ARM Compute Library)**
- `__xla_cpu_runtime_ACLMatMulF32`
- `__xla_cpu_runtime_ACLBatchMatMulF32`

**Convolution (Eigen)**
- `__xla_cpu_runtime_EigenConv{2D,3D}{F16,F32}`
- `__xla_cpu_runtime_EigenSingleThreadedConv{2D,3D}{F16,F32}`

**Convolution (ARM Compute Library)**
- `__xla_cpu_runtime_ACLConv2DF32`

**Collectives**
- `__xla_cpu_runtime_AllReduce`
- `__xla_cpu_runtime_AllGather`
- `__xla_cpu_runtime_ReduceScatter`
- `__xla_cpu_runtime_AllToAll`
- `__xla_cpu_runtime_CollectivePermute`

**Infeed / Outfeed**
- `__xla_cpu_runtime_AcquireInfeedBufferForDequeue`
- `__xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue`
- `__xla_cpu_runtime_AcquireOutfeedBufferForPopulation`
- `__xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation`

**Utilities**
- `__xla_cpu_runtime_ParallelForkJoin` -- multi-threaded parallel regions
- `__xla_cpu_runtime_KeyValueSort` -- sorting
- `__xla_cpu_runtime_TopKF32` -- top-K selection
- `__xla_cpu_runtime_PrintfToStderr` -- debug printing
- `__xla_cpu_runtime_StatusIsSuccess` -- status checking
- `__xla_cpu_runtime_HandleFfiCall` -- FFI dispatch
- `__xla_cpu_runtime_ReplicaId` / `__xla_cpu_runtime_PartitionId` -- SPMD IDs
- `__xla_cpu_runtime_TracingStart` / `__xla_cpu_runtime_TracingEnd` -- profiling

## FunctionLibrary (Typed Symbol Resolution)

The `FunctionLibrary` class (`xla/backends/cpu/runtime/function_library.h`)
provides type-safe resolution of compiled functions at runtime:

```cpp
class FunctionLibrary {
  using Kernel = XLA_CPU_Kernel;      // JIT-compiled fusion kernels
  using Comparator = void(...);       // Sort comparator functions

  template <typename F>
  absl::StatusOr<F*> ResolveFunction(absl::string_view name);
};
```

Each thunk that needs a compiled function (e.g., `KernelThunk` for fusions,
`SortThunk` for comparators) resolves it from the `FunctionLibrary` by name
and type.

## Libraries Linked at Runtime

### Always required

| Library | What it provides | Used by |
|---|---|---|
| **Eigen** (`@eigen_archive//:eigen3`) | GEMM, convolution kernels, thread pool device | MatMul, Conv thunks, thread pool task runners |
| **LLVM** (`@llvm-project//...`) | JIT compilation (ORC JIT), code generation, optimization | `CpuCompiler`, `IrEmitter`, `IrEmitter2` |
| **Abseil** (`@com_google_absl//...`) | Status, containers, synchronization, string utilities | Everywhere |
| **TSL** (`@tsl//...`) | Platform abstractions, async values, threading, profiling | Thread pools, async execution, tracing |
| **DUCC** | FFT implementation | `FftThunk` |

### Optional / conditionally linked

| Library | Build flag | What it provides |
|---|---|---|
| **oneDNN** (`@tsl//tsl/mkl:onednn`) | `--define=with_onednn=true` | Optimized fused matmul, convolution, layer norm, softmax |
| **XNNPACK/YNNPACK** (`@XNNPACK//ynnpack`) | Enabled by default on supported platforms | Mobile/edge-optimized neural network kernels |
| **ARM Compute Library (ACL)** | ARM builds | ARM-optimized matmul and convolution |
| **MPI** | `--define=with_mpi=true` | Distributed collectives via MPI |
| **Gloo** | `--define=with_gloo=true` | Distributed collectives via Gloo |
| **Slinky ThreadPool** (`@slinky//...`) | With XNNPACK | Alternative thread pool for XNNPACK operations |

### Dependency graph (simplified)

```
pjrt_c_api_cpu_plugin.so
  |
  +-- PjRtCpuClient
  |     +-- CpuCompiler
  |     |     +-- LLVM JIT (ORC)
  |     |     +-- ThunkEmitter
  |     |     +-- IrEmitter / IrEmitter2
  |     |     +-- oneDNN pattern matchers (optional)
  |     |
  |     +-- ThunkExecutor
  |           +-- KernelThunk ---------> LLVM-compiled native code
  |           +-- DotThunk ------------> Eigen GEMM / oneDNN matmul
  |           +-- ConvolutionThunk ----> Eigen conv / oneDNN conv / ACL
  |           +-- FftThunk ------------> DUCC
  |           +-- CollectiveThunks ----> In-process / Gloo / MPI
  |           +-- OneDnnFusionThunk --> oneDNN fused ops (optional)
  |           +-- YnnFusionThunk -----> XNNPACK fused ops (optional)
  |           +-- CustomCallThunk ----> User code / FFI handlers
  |
  +-- Eigen::ThreadPoolDevice (intra-op parallelism)
  +-- TaskRunner (inter-thunk parallelism)
```

## Collectives

CPU collectives are abstracted behind the `CpuCollectives` interface
(`xla/backends/cpu/collectives/cpu_collectives.h`). Three backends exist:

| Backend | File | Use case |
|---|---|---|
| **In-process** | `in_process_collectives.cc` | Single-machine, multi-device (default) |
| **Gloo** | `gloo_collectives.cc`, `gloo_communicator.cc` | Distributed training (TCP/IP) |
| **MPI** | `mpi_collectives.cc`, `mpi_communicator.cc` | Distributed training (MPI fabric) |

Each backend implements the `Communicator` interface supporting all-reduce,
all-gather, reduce-scatter, all-to-all, and collective-permute. Rendezvous
and clique management coordinate multi-participant execution.

## Custom Calls and FFI

Two mechanisms exist for user-defined operations:

### Untyped Custom Calls (legacy, API versions 1-3)

```cpp
void MyCustomCall(void* output, const void** inputs,
                  const char* opaque, size_t opaque_len,
                  XlaCustomCallStatus* status);
```

Registered via `XLA_REGISTER_CUSTOM_CALL_TARGET` or loaded from a dynamic
library at runtime.

### FFI Custom Calls (API version 4+)

Type-safe foreign function interface via `xla::ffi::HandlerRegistration`.
CPU-specific FFI context includes:
- `IntraOpThreadPool` -- access to the Eigen thread pool device

The FFI dispatch goes through `__xla_cpu_runtime_HandleFfiCall`.

## Threading Model

XLA:CPU uses two levels of parallelism:

1. **Inter-thunk parallelism** -- The `ThunkExecutor` schedules independent
   thunks concurrently via a `TaskRunner`. An `ExecuteSession` limits
   concurrent workers (default: 4 workers, split threshold: 8 ready thunks).

2. **Intra-op parallelism** -- Individual operations (matmul, convolution,
   element-wise kernels) use Eigen's `ThreadPoolDevice` for data-parallel
   execution within a single operation.

## CPU Client Configuration

`CpuClientOptions` (in `xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h`)
controls runtime behavior:

| Option | Purpose |
|---|---|
| `cpu_device_count` | Number of logical CPU devices |
| `asynchronous` | Enable async dispatch of computations |
| `max_inflight_computations_per_device` | Backpressure limit |
| `process_id` | Process ID for distributed collectives |
| `collectives` | Collectives implementation (in-process, Gloo, MPI) |
| `allocator` | Custom memory allocator |
| `max_transpose_threads` | Thread limit for transpose operations |
| `topology` | CPU topology description |
| `customize_hlo_module_config` | Callback to customize HLO module config |

## Compiler Pipeline (brief)

The `CpuCompiler` transforms HLO through these major phases:

1. **HLO optimization passes** -- Standard simplifications, layout assignment,
   fusion decisions, oneDNN/XNNPACK pattern matching and rewriting.
2. **Thunk emission** (`ThunkEmitter`) -- Converts optimized HLO operations
   into `Thunk` objects. Complex ops become specialized thunks (DotThunk,
   ConvolutionThunk, etc.).
3. **Kernel compilation** (`IrEmitter2`) -- Fusions and element-wise ops are
   lowered to LLVM IR, then JIT-compiled to native code. The resulting
   function pointers are stored in a `FunctionLibrary`.
4. **CPU feature detection** -- `TargetMachineFeatures` detects ISA extensions
   (AVX, AVX2, AVX-512, AMX, etc.) to guide code generation.
