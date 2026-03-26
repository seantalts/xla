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

## Runtime Libraries Required at Execution Time

When an AOT-compiled binary runs, the thunks dispatch to external libraries.
These are the libraries that must be **linked into (or shipped with) the
runtime**, and whose versions must be compatible.

### Core (always required)

| Library | Bazel target | Called by | Versioning concern |
|---|---|---|---|
| **Eigen** | `@eigen_archive//:eigen3` | `DotThunk`, `ConvolutionThunk`, `Eigen::ThreadPoolDevice` | ABI: Eigen tensor contraction interface, thread pool device API. Changes to GEMM dispatch or tensor layout break compatibility. |
| **DUCC** | `@ducc` | `FftThunk` | ABI: FFT function signatures. Relatively stable. |
| **Abseil** | `@com_google_absl//...` | Everywhere (status, containers, sync) | ABI: Abseil has inline namespaces for ABI versioning. Must match between compile-time and runtime. |
| **TSL** | `@tsl//...` | `ThunkExecutor` (async values), task runners, tracing | ABI: `AsyncValueRef`, thread pool interfaces. |
| **Protobuf** | `@com_google_protobuf` | Deserialization of `CompilationResultProto`, all thunk protos | ABI: Proto wire format is stable, but generated C++ code has ABI tied to protobuf version. |

Note: **LLVM is NOT required at runtime** for AOT binaries. LLVM is only used
at compile time. The object files in the artifact contain fully compiled
native code.

### Conditional (depends on what ops the model uses)

| Library | Bazel target | Called by | When needed |
|---|---|---|---|
| **oneDNN** | `@tsl//tsl/mkl:onednn` | `OneDnnFusionThunk` (fused matmul, conv, layernorm, softmax) | When compiler emits oneDNN fusions (currently disabled for AOT) |
| **XNNPACK/YNNPACK** | `@XNNPACK//ynnpack` | `YnnFusionThunk` | When compiler emits XNNPACK fusions |
| **ARM Compute Library** | ACL | `DotThunk`, `ConvolutionThunk` (ARM path) | ARM builds only |
| **MPI** | system MPI | `CollectiveThunks` via `MpiCommunicator` | Multi-node with MPI collectives |
| **Gloo** | `@gloo` | `CollectiveThunks` via `GlooCommunicator` | Multi-node with Gloo collectives |
| **Slinky ThreadPool** | `@slinky//slinky/base:thread_pool` | `YnnFusionThunk` | When XNNPACK is used |

### What each thunk type calls at runtime

| Thunk | External library call |
|---|---|
| `KernelThunk` | **None** -- calls into the compiled object file via `FunctionLibrary` |
| `DotThunk` | **Eigen** -- `dot_lib_{f16,f32,f64,c64,c128,s32,s8}.cc` via Eigen tensor contraction |
| `ConvolutionThunk` | **Eigen** -- `convolution_lib_{f16,f32}_{2d,3d}.cc` via `eigen_spatial_convolutions.h` |
| `FftThunk` | **DUCC** -- `ducc/google/fft.h` |
| `SortThunk` | **C++ STL** -- `std::sort` / `std::stable_sort` + compiled comparator from `FunctionLibrary` |
| `TopKThunk` | **C++ STL** -- `std::partial_sort` |
| `CopyThunk` | **None** -- `memcpy` |
| `CustomCallThunk` | **User code** -- registered custom call targets or FFI handlers |
| `CollectiveThunks` | **Collectives backend** -- in-process, Gloo, or MPI |
| `OneDnnFusionThunk` | **oneDNN** -- `dnnl_graph.hpp` |
| `YnnFusionThunk` | **XNNPACK** -- via `ynn_interop.h` |
| `InfeedThunk`/`OutfeedThunk` | **XfeedManager** -- internal queue implementation |
| `WhileThunk`/`CallThunk`/`ConditionalThunk` | **None** -- control flow over nested thunk sequences |
| `RngGetAndUpdateStateThunk` | **None** -- internal state update |
| `ReplicaIdThunk`/`PartitionIdThunk` | **None** -- reads from execution context |

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

| Library | What could break |
|---|---|
| **Eigen** | Tensor contraction ABI, `ThreadPoolDevice` API, GEMM dispatch changes |
| **DUCC** | FFT function signatures |
| **oneDNN** | Graph API (`dnnl_graph.hpp`) version |
| **XNNPACK** | Subgraph/runtime API version |
| **Protobuf** | C++ generated code ABI (inline namespace changes between major versions) |

### Summary: versioning priority

1. **`kernel_c_api.h` structs** -- Highest priority. Baked into compiled code. Must be versioned or frozen.
2. **`ThunkSequenceProto` schema** -- Proto evolution rules help, but need explicit version tracking.
3. **`CompilationResultProto` schema** -- Container format, same concerns as thunk protos.
4. **Thunk semantic contracts** -- Harder to version. Requires integration testing.
5. **External library versions** -- Eigen, DUCC, oneDNN, XNNPACK must be pinned or compatibility-tested.
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
