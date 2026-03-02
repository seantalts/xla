# XLA CPU Performance Analysis: JAX Issue #26021

## Issue Summary

JAX users report a **3-5x regression** in CPU execution time for robotics
workloads (MuJoCo JAX / MJX) starting with JAX 0.4.32, persisting through
JAX 0.9.0. The regression correlates with the introduction of the thunk-based
CPU runtime.

**Benchmark numbers (mass_matrix on Unitree G1 robot, f64, CPU):**
| JAX Version | Avg Time | Relative |
|------------|----------|----------|
| 0.4.30     | 0.023 ms | 1.0x     |
| 0.4.33     | 0.133 ms | 5.8x     |
| 0.9.0      | 0.087 ms | 3.8x     |

## Workload Characteristics

MJX workloads have properties that make them especially sensitive to per-op
overhead:

1. **Many small operations**: Kinematics chains produce 50-200+ HLO
   instructions, each operating on tiny tensors (6x6 spatial inertia matrices,
   36-element joint vectors). Actual compute per op is ~10-100ns.

2. **Sequential dependencies**: Kinematic chains are inherently sequential -
   each joint's transform depends on the previous. This means the ThunkExecutor
   runs in `is_sequential_` mode.

3. **F64 precision**: Robotics requires double precision, which means no
   vectorized SIMD speedups on small tensors.

4. **Custom calls via FFI**: MJX uses JAX's FFI for specialized physics
   computations, adding FFI dispatch overhead per call.

## Identified Overhead Sources

### 1. FFI Custom Call Dispatch (`custom_call_thunk.cc:298-349`)

**Per-call overhead in `CallTypedFFI`:**

```
Hot path per FFI call:
  a) Allocate InlinedVector<DeviceAddressBase, 8> for arguments   [~20ns]
  b) For each arg: GetDeviceAddress(slice) → StatusOr             [~15ns/arg]
  c) Allocate InlinedVector<DeviceAddressBase, 4> for results     [~20ns]
  d) For each ret: GetDeviceAddress(slice) → StatusOr             [~15ns/ret]
  e) ObjectPool::GetOrCreate() → CAS + potential Copy()           [~30-50ns]
  f) CallFrame::UpdateWithBuffers()                               [~10ns]
  g) Construct ffi::CallOptions (with std::variant)               [~15ns]
  h) ffi::CallAsync() → CreateExecutionContext() + Build()        [~40ns]
     → try-catch wrapper → handler call → TakeFuture()
  Total: ~200-300ns per FFI call
```

The old runtime dispatched custom calls with a direct function pointer call
(~5-10ns), so this represents a ~20-30x overhead per custom call.

**Key concern:** `CreateExecutionContext()` in `ffi_api.cc:86-121` uses
`std::visit(BackendVisitor{}, options.backend_options)` on every call. This
variant dispatch is unnecessary since the backend type is known at compile time
for CPU.

### 2. ThunkExecutor Sequential Dispatch (`thunk_executor.cc:308-378`)

For MJX workloads, `is_sequential_ = true` (small buffers < 512 bytes), so
`ExecuteSequential` is used. Per-thunk overhead:

```
Per thunk in sequential path:
  a) TracedExecute: check TraceMe::Active()                       [~5ns]
  b) thunk.Execute() virtual dispatch                             [~5ns]
  c) AsyncValueRef creation/checking                              [~10-15ns]
  d) thunk.IsOkExecuteEvent() check                              [~5ns]
  Total: ~25-30ns per thunk (excluding actual compute)
```

With 100+ thunks, this adds ~3us per execution, which is significant when
total execution is ~23us (JAX 0.4.30 baseline).

### 3. CallFrame ObjectPool (`object_pool.h`)

The `ObjectPool<CallFrame>` uses lock-free CAS operations:
- First call: allocates + copies call frame (~500ns)
- Steady state: PopEntry() CAS + PushEntry() CAS (~30-50ns per borrow/return)
- The `BorrowedObject` destructor returns the frame to the pool

While the pool is well-optimized, the Copy() path for CallFrame allocates:
- `make_unique<Arguments>` (copies vector of buffers)
- `make_unique<Results>` (copies vector of buffers)
- Runs `FixUpArgs` and `FixUpRets` to fix internal pointers

### 4. StatusOr Overhead on Hot Path

Multiple functions on the hot path return `absl::StatusOr<T>` which has
non-trivial overhead vs returning T directly:

- `GetDeviceAddress(slice)` → `StatusOr<DeviceAddressBase>` (called per buffer)
- `call_frames_.GetOrCreate()` → `StatusOr<BorrowedObject>`
- `CallAsync()` → `StatusOr<XLA_FFI_Future*>` internally

In an inner loop running 100+ times per execution, these add up.

### 5. BufferAllocations::GetDeviceAddress

Each buffer lookup does bounds checking and offset computation:
```cpp
StatusOr<DeviceAddressBase> GetDeviceAddress(
    const BufferAllocation::Slice& buffer_slice) const;
```

For the kernel thunk, there's an optimized unchecked path
(`GetDeviceAddressUnchecked`) used when `ShouldCheckBufferSlices()` is false.
The custom call thunk does NOT use this optimization.

### 6. AsyncValue Infrastructure

Every thunk execution creates and returns `AsyncValueRef<ExecuteEvent>`.
Even for synchronous operations that complete immediately:
- `OkExecuteEvent()` returns a pre-allocated static async value (good)
- But the checking path (`IsOkExecuteEvent`, `IsAvailable`) still has overhead

## Recommendations

### High Impact (Expected: 2-3x improvement)

1. **Add unchecked buffer address path for CustomCallThunk**
   Like `KernelThunk`, skip bounds checking in release builds:
   ```cpp
   // Instead of:
   TF_ASSIGN_OR_RETURN(arguments.emplace_back(),
                        params.buffer_allocations->GetDeviceAddress(slice));
   // Use:
   arguments.emplace_back(
       params.buffer_allocations->GetDeviceAddressUnchecked(slice));
   ```

2. **Cache ExecutionContext per-execution instead of per-call**
   `CreateExecutionContext()` creates the same context for every FFI call in a
   single execution. It should be created once in `ExecuteThunks()` and passed
   through `ExecuteParams`:
   ```cpp
   // In cpu_executable.cc ExecuteThunks():
   XLA_FFI_ExecutionContext ffi_ctx = CreateExecutionContext(options);
   // Pass ffi_ctx through execute_params to avoid recreating per custom call
   ```

3. **Avoid std::visit for backend context on CPU**
   Since the CPU backend always uses `CpuOptions`, the variant dispatch is
   unnecessary. Use a direct struct access or template specialization.

4. **Pre-build XLA_FFI_CallFrame in CallFrame**
   Instead of calling `Build()` on every invocation, cache the
   `XLA_FFI_CallFrame` struct and only update the `ctx` pointer.

### Medium Impact (Expected: 20-50% improvement)

5. **Remove try-catch from FFI hot path for internal handlers**
   For statically-linked handlers (which is the common case for CPU), the
   try-catch in `ffi_api.cc:176-186` is unnecessary. Use a separate dispatch
   path for internal handlers.

6. **Inline OkExecuteEvent check in ExecuteSequential**
   The `IsOkExecuteEvent` check can be made branchless or combined with the
   `IsAvailable` check.

7. **Consider templating CustomCallThunk on buffer count**
   Like `SmallKernelThunk<N,M>`, specialize for common buffer counts to
   eliminate dynamic vector allocation.

### Low Impact (Cleanup)

8. **Profile with `perf` and trace with `--xla_dump_to`**
   Run with `XLA_FLAGS=--xla_dump_to=/tmp/xla_dump` to capture HLO and
   identify the exact operation mix. Use `perf record` on the benchmark
   to confirm overhead distribution.

## How to Profile

### Build and run the MJX-like benchmark:
```bash
bazel run -c opt //xla/backends/cpu/benchmarks:many_small_ops_benchmark_test \
  -- --benchmark_filter=".*CustomCallChain.*"
```

### Profile with perf:
```bash
bazel build -c opt //xla/backends/cpu/benchmarks:many_small_ops_benchmark_test

perf record -g ./bazel-bin/xla/backends/cpu/benchmarks/many_small_ops_benchmark_test \
  --benchmark_filter="BM_CustomCallChain/50" --benchmark_min_time=5s

perf report --no-children
```

### Profile with XLA tracing:
```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" \
  ./bazel-bin/xla/backends/cpu/benchmarks/many_small_ops_benchmark_test \
  --benchmark_filter="BM_InterleavedComputeAndCustomCalls/25"
```

### Compare with blocking executor (shows per-thunk timing):
```bash
bazel run -c opt --copt=-DXLA_CPU_USE_BLOCKING_THUNK_EXECUTOR \
  //xla/backends/cpu/benchmarks:many_small_ops_benchmark_test \
  -- --benchmark_filter=".*" -v=2
```

## Files of Interest

| File | Relevance |
|------|-----------|
| `xla/backends/cpu/runtime/custom_call_thunk.cc` | FFI dispatch hot path |
| `xla/backends/cpu/runtime/thunk_executor.cc` | Sequential execution loop |
| `xla/ffi/ffi_api.cc` | CreateExecutionContext, Call/CallAsync |
| `xla/ffi/call_frame.cc` | UpdateWithBuffers, Copy, Build |
| `xla/runtime/object_pool.h` | Lock-free call frame pooling |
| `xla/backends/cpu/runtime/kernel_thunk.cc` | Optimized kernel dispatch (reference) |
| `xla/service/cpu/cpu_executable.cc` | ExecuteThunks entry point |
