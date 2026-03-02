# XLA CPU Performance Analysis: JAX Issue #26021

## Issue Summary

JAX users report a **2-3x regression** in CPU execution time for small-op
workloads starting with JAX 0.4.32, persisting through JAX 0.6.0. The
upstream fix ([xla:cpu] Resolve custom call target at construction time)
achieved only ~9% improvement, leaving the majority of the regression
unresolved.

**Key observations from the issue:**
- 2-3x slowdown on CPU; GPU unaffected
- `--xla_cpu_use_thunk_runtime=false` made things **worse**, not better
- The workload is many small QP (quadratic program) solves via qpax/FFI

## Root Cause Analysis: It's NOT (just) the Thunk Runtime

Our initial analysis focused on thunk-internal overhead (FFI dispatch,
buffer address checking, `std::visit` in `CreateExecutionContext`). While
those are real costs, they account for only ~9% of the regression (matching
the upstream fix's improvement). The **dominant overhead is in the PjRt
execution scaffolding** — the layers between JAX's Python dispatch and XLA's
thunk execution.

## Per-Execution Overhead Budget

For every single JAX operation on CPU, the following work happens before
any actual computation:

### Layer 1: CommonPjRtClient PrepareArguments (~1-3μs)

**File:** `xla/pjrt/common_pjrt_client.cc:522-684`

For each input buffer:
```
GetBufferWithHold(kUsage):
  absl::MutexLock lock(mu_)          [~50-100ns per buffer]
  WaitForOutstandingDonationHold()
  AcquireHoldLocked()

Shape validation (if strict_shape_checking)  [~10ns per buffer]
TestBufferDonationClashes()                  [~10ns per buffer]
AddDefinitionEventsToSet()                   [~10-20ns per buffer]
```

For 4 input buffers: ~300-500ns just in mutex locks + event tracking.

### Layer 2: Output Buffer Allocation (~0.5-1μs)

**File:** `xla/pjrt/common_pjrt_client.cc:686-808`

```
AllocateOutputBuffersWithInputReuse():
  Per output: memory space lookup, alias check, buffer allocation
  AllocateRawBufferForExecute() with OOM retry logic
```

### Layer 3: CpuPjRtRawLoadedExecutable::Execute (~2-5μs)

**File:** `xla/pjrt/cpu/cpu_client.cc:1491-1827`

```
MakeConstructedAsyncValueRef<CpuEvent>() x2         [~50ns each]
CreateBufferTable() — iterates all allocations       [~100-500ns]
  Per allocation: MemoryForAllocation()
  Handles params, constants, temps, outputs

Semaphore::ScopedAcquire(1)                          [~100-300ns]
  absl::Mutex::LockWhen(Condition) + unlock
  NOTE: has TODO "Optimize semaphore related overhead"

ExecutableRunOptions setup                           [~50ns]
CpuExecutableRunOptions allocation (make_unique)     [~50ns]
CollectiveExecuteParams::Create()                    [~20ns]
CustomCallExecuteParams::Create()                    [~20ns]
ThreadPoolTaskRunner construction                    [~20ns]

Buffer table → buffer_device_mem conversion loop     [~50-100ns]
BufferAllocations construction                       [~20ns]

Thunk::ExecuteParams construction                    [~20ns]

ScopedFlushDenormal + ScopedSetRound                 [~20ns]
```

### Layer 4: Post-Execution (~0.5-2μs)

**File:** `xla/pjrt/common_pjrt_client.cc:967-998`

```
For each input buffer:
  ConvertUsageHold():
    absl::MutexLock lock(parent()->mu_)    [~50-100ns per buffer]
    AddUsageEvent()
    DecrementUsage()

Semaphore::Release(1)                      [~100ns]
  absl::MutexLock + value_ += amount

AsyncValueRef SetStateConcrete()           [~20ns]
```

### Layer 5: Thunk Execution (~0.5-3μs for thunk overhead)

**File:** `xla/backends/cpu/runtime/thunk_executor.cc`

```
Per thunk (sequential path):
  TracedExecute()                          [~5ns]
  thunk.Execute() virtual dispatch         [~5ns]
  AsyncValueRef check                      [~10ns]
  Total: ~20-25ns × N thunks
```

For 100 thunks: ~2-2.5μs in dispatch overhead.

### Total Per-Execution Overhead

| Component | Estimated Cost | Notes |
|-----------|---------------|-------|
| Buffer hold acquisition | 300-500ns | N × mutex lock per input |
| Output allocation | 500-1000ns | Memory alloc per output |
| CreateBufferTable | 100-500ns | Per buffer-assignment alloc |
| Semaphore acquire+release | 200-600ns | 2 mutex operations |
| AsyncValue creation | 100-200ns | 2-3 async values per exec |
| Post-exec hold conversion | 300-500ns | N × mutex lock per input |
| Thunk dispatch overhead | 500-2500ns | 20-25ns × N thunks |
| **Total overhead** | **2-6μs** | **Before any real work** |

For a computation that should take ~23μs (JAX 0.4.30 baseline), adding
4-6μs of overhead = 1.2-1.3x slowdown from PjRt alone. Combined with
thunk runtime overhead, this could explain the 2-3x regression.

## Why the Thunk Runtime Flag Didn't Help

The user reported `--xla_cpu_use_thunk_runtime=false` made things worse.
Current code at `cpu_client.cc:1662` shows:

```cpp
if (!cpu_executable->has_thunks()) {
    return Internal("CpuExecutable has no thunks.");
}
```

The non-thunk path no longer exists in the PjRt CPU client. The flag may
affect compilation but the execution path requires thunks.

## What Changed Between 0.4.31 and 0.4.32

The regression is likely a combination of:

1. **Thunk runtime became default** — added per-thunk dispatch overhead
   (~20-25ns/thunk) that didn't exist with direct function calls

2. **PjRt execution machinery grew** — more event tracking, async
   execution tracking (CpuAsyncExecutionTracker), stream event maps

3. **Buffer management added more safety** — mutex-based holds, donation
   tracking, event chains for correctness

4. **FFI dispatch overhead** — thunk-based FFI dispatch is heavier than
   the old direct function pointer call

## Recommendations

### High Impact: PjRt Layer

1. **Skip semaphore for cheap computations** — The `cheap_computation_`
   flag already exists (line 1631). Use it to skip semaphore acquire:
   ```cpp
   if (!executable_->cheap_computation_) {
     compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
         device_->max_inflight_computations_semaphore().ScopedAcquire(1));
   }
   ```

2. **Fast-path for synchronous execution** — When `execute_inline` is
   true, many async tracking structures are unnecessary. Skip:
   - `CpuAsyncExecutionTracker::NewAsyncExecution()`
   - `ExecutionStreamEventMap` tracking
   - Extra `AsyncValueRef<CpuEvent>` creation

3. **Reduce per-buffer mutex overhead** — For `ConvertUsageHold`, batch
   the mutex acquisition or use lock-free event tracking for usage events.

4. **Pre-allocate buffer table** — `CreateBufferTable` creates a new
   `vector<AsyncValueRef>` each execution. For repeated execution of the
   same program, reuse the table.

### Medium Impact: Thunk Layer

5. **Unchecked buffer addresses in CustomCallThunk** — ✅ Already done in
   this branch. Matches `KernelThunk` pattern using
   `ShouldCheckBufferSlices()`.

6. **Cache ExecutionContext per-execution** — Create
   `XLA_FFI_ExecutionContext` once in `ExecuteThunks` and pass through
   `ExecuteParams`, avoiding reconstruction per custom call.

7. **InvokeAsync with pre-built context** — Main's `InvokeContext` is
   already cleaner, but still uses `std::visit` internally. For the CPU
   backend, directly construct `XLA_FFI_ExecutionContext::CpuContext`.

### Low Impact: Thunk Internals

8. **Remove try-catch for internal handlers** — For statically-linked
   handlers, the exception guard adds ~5-10ns overhead.

9. **Template CustomCallThunk on buffer count** — Like
   `SmallKernelThunk<N,M>`, avoid dynamic allocation for common cases.

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

### Profile the FULL execution path (PjRt + XLA):
```bash
# Use JAX profiler to capture the complete picture:
import jax
jax.profiler.start_trace("/tmp/jax_trace")
# ... run workload ...
jax.profiler.stop_trace()
# View in TensorBoard
```

### Compare with blocking executor (shows per-thunk timing):
```bash
bazel run -c opt --copt=-DXLA_CPU_USE_BLOCKING_THUNK_EXECUTOR \
  //xla/backends/cpu/benchmarks:many_small_ops_benchmark_test \
  -- --benchmark_filter=".*" -v=2
```

## Files of Interest

| File | Layer | Relevance |
|------|-------|-----------|
| `xla/pjrt/common_pjrt_client.cc` | PjRt | PrepareArguments, buffer holds |
| `xla/pjrt/abstract_tracked_device_buffer.cc` | PjRt | Per-buffer mutex locks |
| `xla/pjrt/cpu/cpu_client.cc` | PjRt/CPU | Execute path, semaphore, buffer table |
| `xla/pjrt/cpu/cpu_device.h` | PjRt/CPU | Semaphore TODO comment |
| `xla/pjrt/semaphore.h` | PjRt | Mutex-based semaphore |
| `xla/pjrt/cpu/cpu_async_execution_tracker.h` | PjRt/CPU | Async tracking overhead |
| `xla/backends/cpu/runtime/custom_call_thunk.cc` | Thunk | FFI dispatch hot path |
| `xla/backends/cpu/runtime/thunk_executor.cc` | Thunk | Sequential execution loop |
| `xla/ffi/invoke.cc` | FFI | InvokeAsync, CreateExecutionContext |
| `xla/ffi/call_frame.cc` | FFI | UpdateWithBuffers, Copy, Build |
| `xla/backends/cpu/runtime/kernel_thunk.cc` | Thunk | Reference: optimized dispatch |
| `xla/service/cpu/cpu_executable.cc` | XLA | ExecuteThunks entry point |
