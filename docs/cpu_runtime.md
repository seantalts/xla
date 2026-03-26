# XLA:CPU AOT Runtime Surface Area

This document describes the runtime surface area of an AOT-compiled XLA:CPU
binary: what libraries must be present at runtime, what ABIs the compiled code
depends on, and what would need to be versioned to allow an AOT artifact to
work across different versions of the XLA runtime.

## How AOT Compilation Works

AOT compilation produces a `CompilationResultProto` (defined in
`xla/service/cpu/executable.proto`) containing:

```protobuf
message CompilationResultProto {
  HloModuleProtoWithConfig hlo_module = 1;      // Original program metadata
  BufferAssignmentProto buffer_assignment = 2;   // Static buffer layout
  string entry_function_name = 3;                // Entry point symbol
  ObjFileKind obj_files_kind = 5;                // CLASSIC or KERNELS mode
  ThunkSequenceProto thunk_sequence = 6;         // Serialized execution plan
  repeated SymbolProto compiled_symbols = 7;     // Kernel/comparator symbols
  repeated ObjFileProto object_files = 8;        // Compiled .o files
  TargetMachineOptionsProto target_machine_options = 9;  // triple, cpu, features
}
```

At load time, the runtime:
1. Loads the object files and links the compiled symbols into a `FunctionLibrary`
2. Deserializes the `ThunkSequenceProto` back into a `ThunkSequence`
3. Creates a `ThunkExecutor` that runs the thunks against the function library

This means the compiled artifact depends on **two distinct ABI surfaces**: the
**kernel calling convention** (how compiled code is invoked) and the **thunk
runtime** (how operations are orchestrated and what libraries they call into).

## ABI Surface 1: Kernel Calling Convention

Compiled kernels (element-wise ops, fusions) follow the C ABI defined in
`xla/backends/cpu/runtime/kernel_c_api.h`:

```c
typedef struct XLA_CPU_KernelArg {
  void* data;
  size_t size;
} XLA_CPU_KernelArg;

typedef struct XLA_CPU_KernelCallFrame {
  const XLA_CPU_NumWorkGroups* num_workgroups;  // {x, y, z}
  const XLA_CPU_WorkGroupId* workgroup_id;      // {x, y, z}
  size_t num_args;
  const XLA_CPU_KernelArg* args;
} XLA_CPU_KernelCallFrame;

// Every compiled kernel has this signature:
typedef XLA_CPU_KernelError* XLA_CPU_Kernel(
    const XLA_CPU_KernelCallFrame* call_frame);
```

Sort comparators use a different (legacy) signature:

```c
void Comparator(bool* result, const void* run_options,
                const void** params, const void* buffer_table,
                const void* status, const void* prof_counters);
```

**Versioning implications**: Any change to `XLA_CPU_KernelCallFrame`,
`XLA_CPU_KernelArg`, or the comparator signature is a breaking ABI change.
AOT artifacts compiled against one version of these structs cannot be used
with a runtime that uses a different layout.

## ABI Surface 2: Thunk Serialization Format

The thunk sequence is serialized as `ThunkSequenceProto`
(`xla/backends/cpu/runtime/thunk.proto`). Each thunk type has its own proto:

| Proto Message | Thunk | What it encodes |
|---|---|---|
| `KernelThunkProto` | `KernelThunk` | kernel name, workgroup dims, buffer slices |
| `DotThunkProto` | `DotThunk` | dot dimensions, LHS/RHS/output buffer shapes |
| `ConvolutionThunkProto` | `ConvolutionThunk` | window, dimension numbers, feature groups |
| `FftThunkProto` | `FftThunk` | FFT type, lengths, threading mode |
| `SortThunkProto` | `SortThunk` | comparator name, dimension, stability |
| `TopKThunkProto` | `TopKThunk` | batch size, input size, k |
| `CopyThunkProto` | `CopyThunk` | src/dst buffer shapes |
| `CollectiveThunkProto` | Various | op params, buffers, resources + per-type config |
| `CustomCallThunkProto` | `CustomCallThunk` | target name, API version, backend config |
| `CallThunkProto` | `CallThunk` | nested thunk sequence |
| `ConditionalThunkProto` | `ConditionalThunk` | branch sequences, branch index buffer |
| `WhileThunkProto` | `WhileThunk` | cond/body sequences, trip count |
| `InfeedThunkProto` | `InfeedThunk` | buffers, token resources |
| `OutfeedThunkProto` | `OutfeedThunk` | buffers, token resources |
| `YnnFusionThunkProto` | `YnnFusionThunk` | XNNPACK fusion options, instruction id |
| `RngGetAndUpdateStateThunkProto` | `RngGetAndUpdateStateThunk` | delta, state buffer |
| `PartitionIdThunkProto` | `PartitionIdThunk` | logical id buffer |
| `ReplicaIdThunkProto` | `ReplicaIdThunk` | logical id buffer |

**Versioning implications**: The runtime must be able to deserialize every
thunk proto that was produced by the compiler. Adding new thunk types is
backwards-compatible (old artifacts don't use them). Changing or removing
fields in existing protos is a breaking change. The `BufferAssignmentProto`
and `BufferAllocationSliceProto` formats are also part of this contract.

## Runtime Library Dependencies

An important distinction: most of what thunks call is **compiled into the XLA
runtime binary itself**, not dynamically linked at runtime. The AOT artifact
(the `CompilationResultProto`) contains compiled `.o` files with kernel code,
and the *runtime* binary contains compiled thunk implementations. The
question is what external shared libraries the runtime binary itself needs.

### Compiled into the runtime (no separate runtime dependency)

These are header-only or statically-compiled libraries. Their code is baked
into the XLA runtime binary at build time. There is no separate `.so` to
ship, but the version compiled into the runtime determines behavior:

| Library | How it's compiled in | Used by |
|---|---|---|
| **Eigen** | Header-only. Templates are instantiated in `dot_lib_*.cc` and `convolution_lib_*.cc` via `extern template`. | `DotThunk` (GEMM), `ConvolutionThunk` (spatial convolutions), `Eigen::ThreadPoolDevice` (intra-op threading) |
| **DUCC** | Header/source compiled into `fft_thunk.cc`. | `FftThunk` |
| **C++ STL** | Standard library, always present. | `SortThunk` (`std::sort`), `TopKThunk` (`std::partial_sort`) |
| **Abseil** | Statically linked into the runtime binary. | Everywhere (status, containers, synchronization) |
| **TSL** | Statically linked into the runtime binary. | `ThunkExecutor` (async values), task runners, tracing |

The Eigen case is worth highlighting: `Eigen::ThreadPoolDevice*` is passed
into thunks at execution time as a pointer, but the ThreadPoolDevice
implementation is all inlined/compiled into the runtime. The *caller* must
provide a compatible `ThreadPoolDevice` instance (i.e., one built against
the same Eigen headers), but there is no Eigen shared library to link.

### Actual runtime linking dependencies

These are libraries that may exist as separate shared objects (`.so`) that
the runtime binary dynamically links against:

| Library | Bazel target | When needed | Versioning concern |
|---|---|---|---|
| **Protobuf** | `@com_google_protobuf` | Always (deserializing `CompilationResultProto`) | Proto wire format is stable, but C++ generated code ABI is tied to protobuf major version. If protobuf is dynamically linked, versions must match. |
| **oneDNN** | `@tsl//tsl/mkl:onednn` | When `OneDnnFusionThunk` is used (currently disabled for AOT) | `dnnl_graph.hpp` graph API version must match |
| **XNNPACK/YNNPACK** | `@XNNPACK//ynnpack` | When `YnnFusionThunk` is used | XNNPACK subgraph/runtime API version must match |
| **ARM Compute Library** | ACL | ARM builds with ACL-accelerated matmul/conv | ACL API version must match |
| **MPI** | system `libmpi.so` | Multi-node with MPI collectives | MPI ABI (typically stable across minor versions) |
| **Gloo** | `@gloo` | Multi-node with Gloo collectives | Gloo API version |
| **Slinky ThreadPool** | `@slinky//slinky/base:thread_pool` | When XNNPACK is used | Thread pool interface version |

Note: **LLVM is NOT required at runtime**. It is only used at compile time.
The object files in the AOT artifact contain fully compiled native code that
is loaded by the runtime's `FunctionLibrary` without any LLVM involvement.

### What each thunk type calls at runtime

| Thunk | What it calls | Compiled-in or external? |
|---|---|---|
| `KernelThunk` | Compiled kernel from `.o` file via `FunctionLibrary` | Compiled into artifact |
| `DotThunk` | Eigen tensor contraction via `dot_lib_*.cc` | Compiled into runtime |
| `ConvolutionThunk` | Eigen spatial convolution via `convolution_lib_*.cc` | Compiled into runtime |
| `FftThunk` | DUCC FFT via `fft_thunk.cc` | Compiled into runtime |
| `SortThunk` | `std::sort`/`std::stable_sort` + comparator from `FunctionLibrary` | STL + compiled into artifact |
| `TopKThunk` | `std::partial_sort` | STL (compiled into runtime) |
| `CopyThunk` | `memcpy` | libc |
| `CustomCallThunk` | Registered custom call targets or FFI handlers | User-provided (external) |
| `CollectiveThunks` | In-process collectives, or Gloo/MPI | In-process: compiled in. Gloo/MPI: external `.so` |
| `OneDnnFusionThunk` | oneDNN graph API | External (oneDNN `.so`) |
| `YnnFusionThunk` | XNNPACK runtime | External (XNNPACK) |
| `InfeedThunk`/`OutfeedThunk` | `XfeedManager` | Compiled into runtime |
| `WhileThunk`/`CallThunk`/`ConditionalThunk` | Nested thunk sequences | N/A (control flow) |
| `RngGetAndUpdateStateThunk` | Internal state update | Compiled into runtime |
| `ReplicaIdThunk`/`PartitionIdThunk` | Reads from execution context | Compiled into runtime |

## Execution Context (Runtime-Provided State)

At execution time, the runtime provides a `Thunk::ExecuteParams` containing:

| Field | Type | Purpose |
|---|---|---|
| `function_library` | `FunctionLibrary*` | Resolves compiled kernel/comparator symbols from the object files |
| `buffer_allocations` | `BufferAllocations*` | Maps buffer allocation indices to device addresses |
| `intra_op_threadpool` | `Eigen::ThreadPoolDevice*` | Thread pool for intra-op parallelism (matmul, conv, etc.) |
| `task_runner` | `TaskRunner*` | Schedules inter-thunk parallel execution |
| `collectives` | `CpuCollectives*` | Collective communication backend |
| `xfeed` | `XfeedManager*` | Infeed/outfeed queue management |
| `run_id`, `device_id` | IDs | Execution and device identification |
| `ffi_context` | `ffi::ExecutionContext*` | FFI execution state |

**Versioning implications**: The `Thunk::ExecuteParams` struct is the
primary interface between thunks and the runtime. Any field added, removed,
or retyped breaks ABI. The `BufferAllocations` addressing scheme
(index -> address mapping) is also part of this contract.

## User-Facing Execution APIs

### NanoRtExecutable (lightweight, recommended for AOT)

`xla/backends/cpu/nanort/nanort_executable.h` -- minimal API for running
AOT artifacts without the full PJRT stack:

```cpp
auto exe = NanoRtExecutable::Create(compilation_result_proto);

// Caller provides argument, result, and temp buffers:
exe->Execute(arguments, results, temp, options);
```

`ExecuteOptions` requires:
- `Eigen::ThreadPoolDevice*` for models that use Eigen (matmul, conv)
- `ffi::ExecutionContext*` for models that use FFI custom calls

### XlaAotFunction (higher-level wrapper)

`xla/backends/cpu/lite_aot/xla_aot_function.h` -- blocking wrapper around
`NanoRtExecutable` with named arguments/results:

```cpp
auto fn = XlaAotFunction::Create(compilation_result_proto);
fn->set_arg_data("input", input_ptr);
fn->Execute();
void* output = fn->result_data("output");
```

### Full PJRT path

Load the AOT result into a `CpuExecutable` via
`CpuAotCompilationResult::LoadExecutable()`, then run it through the
standard PJRT `Execute` flow.

## What Needs Versioning for Cross-Version Compatibility

To allow an AOT-compiled artifact to work with a different version of the
XLA:CPU runtime, these interfaces need stable versioning:

### 1. Serialization format (`CompilationResultProto`)

| Component | Proto | Stability |
|---|---|---|
| Top-level container | `CompilationResultProto` | Must be wire-compatible. Proto evolution rules apply (don't reuse field numbers, don't change types). |
| Thunk sequence | `ThunkSequenceProto`, `ThunkProto`, all `*ThunkProto` messages | Each thunk proto is a versioning surface. New thunk types are additive. Field changes in existing types are breaking. |
| Buffer assignment | `BufferAssignmentProto`, `BufferAllocationSliceProto` | Buffer indexing scheme must match between compiler and runtime. |
| Symbols | `SymbolProto` | `FunctionTypeId` enum must be stable (KERNEL=1, COMPARATOR=2). |
| Object files | `ObjFileProto` | Raw `.o` bytes -- ABI depends on kernel calling convention, not proto format. |

### 2. Kernel C ABI (`kernel_c_api.h`)

The `XLA_CPU_KernelCallFrame` struct layout is baked into the compiled object
files. Any change to struct sizes, field order, or field types is a hard ABI
break. This is the most critical surface to version.

### 3. Comparator ABI

The sort comparator signature is baked into compiled comparator functions.
Currently uses a legacy calling convention. Changing it breaks AOT sort ops.

### 4. Thunk runtime behavior

Even with stable serialization, the runtime behavior of each thunk type is an
implicit contract:

- **DotThunk** must call Eigen GEMM with the same dimension interpretation
- **ConvolutionThunk** must apply the same padding/stride semantics
- **FftThunk** must produce the same FFT output format
- **CollectiveThunks** must follow the same reduction/permutation semantics

These are semantic contracts, not just binary ABI -- a runtime that computes
different results for the same thunk parameters is just as broken.

### 5. Buffer allocation addressing

The `buffer_assignment` in the proto assigns indices to allocations. The
runtime's `BufferAllocations` must use the same index -> address mapping.
The alignment requirements (`Align()` in `xla/backends/cpu/alignment.h`)
must also match.

### 6. External library ABIs

Only libraries that are **dynamically linked** create a true runtime linking
concern. Libraries compiled into the runtime (Eigen, DUCC, STL) are pinned
at runtime build time and don't need separate versioning -- but the runtime
version itself implicitly pins them.

| Library | Linking | What could break |
|---|---|---|
| **Protobuf** | Often dynamic | C++ generated code ABI (inline namespace changes between major versions) |
| **oneDNN** | Dynamic `.so` | Graph API (`dnnl_graph.hpp`) version |
| **XNNPACK** | Dynamic or static | Subgraph/runtime API version |
| **MPI** | Dynamic `libmpi.so` | MPI ABI (usually stable across minor versions) |
| **Gloo** | Dynamic or static | Transport/algorithm API version |

Eigen, DUCC, and STL are header-only or statically compiled into the runtime
binary. They don't create a separate versioning surface -- their behavior is
determined by which version was compiled into the runtime build.

### Summary: versioning priority

1. **`kernel_c_api.h` structs** -- Highest priority. Baked into compiled `.o` files in the artifact. Must be versioned or frozen.
2. **`ThunkSequenceProto` schema** -- Proto evolution rules help, but need explicit version tracking.
3. **`CompilationResultProto` schema** -- Container format, same concerns as thunk protos.
4. **Thunk semantic contracts** -- Harder to version. Requires integration testing. The runtime's compiled-in Eigen/DUCC version determines matmul/conv/FFT behavior.
5. **Dynamically-linked library versions** -- oneDNN, XNNPACK, Protobuf, MPI/Gloo must be compatible if present.
6. **`BufferAllocations` addressing** -- Index scheme and alignment must match.

## Key Source Locations

| Component | Path |
|---|---|
| Kernel C ABI | `xla/backends/cpu/runtime/kernel_c_api.h` |
| AOT compilation result | `xla/service/cpu/cpu_aot_compilation_result.h` |
| Serialization format | `xla/service/cpu/executable.proto` |
| Thunk protos | `xla/backends/cpu/runtime/thunk.proto` |
| Thunk base class | `xla/backends/cpu/runtime/thunk.h` |
| Thunk executor | `xla/backends/cpu/runtime/thunk_executor.h` |
| Function library | `xla/backends/cpu/runtime/function_library.h` |
| Buffer allocations | `xla/backends/cpu/runtime/buffer_allocations.h` |
| Runtime symbols | `xla/service/cpu/cpu_runtime.h` |
| NanoRt executable | `xla/backends/cpu/nanort/nanort_executable.h` |
| XlaAotFunction | `xla/backends/cpu/lite_aot/xla_aot_function.h` |
| Dot runtime lib | `xla/backends/cpu/runtime/dot_lib.h` |
| Conv runtime lib | `xla/backends/cpu/runtime/convolution_lib.h` |
| FFT thunk | `xla/backends/cpu/runtime/fft_thunk.h` |
| Sort runtime lib | `xla/backends/cpu/runtime/sort_lib.h` |
| CPU FFI bindings | `xla/backends/cpu/ffi.h` |
| Collectives | `xla/backends/cpu/collectives/cpu_collectives.h` |
| oneDNN thunks | `xla/backends/cpu/runtime/onednn/` |
| XNNPACK thunks | `xla/backends/cpu/runtime/ynnpack/` |
| CPU compiler (AOT path) | `xla/service/cpu/cpu_compiler.cc` |
