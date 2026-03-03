# XLA CPU Performance Bisect Instructions

## Issue

[jax-ml/jax#26021](https://github.com/jax-ml/jax/issues/26021) reports a 3-9x CPU performance regression for MuJoCo MJX workloads (mass_matrix function) between JAX 0.4.31 and 0.4.33. The workload consists of ~200 sub-computations with small f64 tensors (input f64[36] → output f64[35,35], ~7000 HLO ops).

## What We Already Know

### Pip-based bisect results (coarse-grained, by jaxlib release)

| JAX/jaxlib | median (us) | vs 0.4.30 |
|---|---|---|
| 0.4.30 | 31.66 | 1.0x (baseline) |
| 0.4.31 | 36.31 | 1.1x |
| 0.4.32 | 279.62 | 8.8x ← REGRESSION |
| 0.4.33 | 294.63 | 9.3x |
| 0.7.0 | 158.21 | 5.0x |
| 0.9.1 | 199.84 | 6.3x |

### Known primary regression commit

**`0a4d157ba44b378221a915965036c0aff0120a70`** (2024-08-07) — "[xla:cpu] Switch XLA:CPU runtime to thunks interpreter" by Eugene Zhulenev. One-line change in `xla/debug_options_flags.cc`:

```diff
-  opts.set_xla_cpu_use_thunk_runtime(false);
+  opts.set_xla_cpu_use_thunk_runtime(true);
```

This is the biggest single regression. But even with `--xla_cpu_use_thunk_runtime=false` on 0.4.33, performance only recovers to ~98us (still 3x worse than 0.4.30's 31.7us), suggesting **additional regression sources**.

### Workaround status

- `--xla_cpu_use_thunk_runtime=false` helps on 0.4.33 but on 0.7.1 (last version with the flag) it's actually 1.5x SLOWER — the thunks runtime was optimized enough by then.
- On 0.7.2+ the flag is deprecated/removed. No workaround exists for current JAX.

## What Still Needs Bisecting

### 1. Fine-grained bisect of the thunks switch (CONFIRMED — optional verification)

XLA commit range: `95e3eea8d2` (0.4.31 pin, 2024-07-28) → `720b2c5334` (0.4.33 pin, 2024-09-10)

The regression commit `0a4d157ba4` is already identified. You can verify this by testing commits immediately before and after.

### 2. The "other 3x" regression — NEEDS INVESTIGATION

Even with thunks disabled, 0.4.33 is 3x slower than 0.4.30. This needs a separate bisect with `--xla_cpu_use_thunk_runtime=false` to isolate non-thunks regressions.

XLA commit range: `79fd5733f9` (0.4.30 pin) → `720b2c5334` (0.4.33 pin)

### 3. Post-0.4.33 improvements and remaining gap — NEEDS INVESTIGATION

Performance improved from 9.3x (0.4.33) to 5.0x (0.7.0) with thunks runtime optimizations. Identify which optimization commits had the most impact to understand what patterns help.

### 4. 0.7.0 → 0.9.1 regression — NEEDS INVESTIGATION

Performance went from 5.0x (0.7.0) to 6.3x (0.9.1). Something regressed again in this range.

XLA commit range for 0.7.0 → 0.9.1 can be found by checking JAX's `third_party/xla/workspace.bzl`.

## HLO File Location

The pre-extracted HLO is at:
```
xla/backends/cpu/benchmarks/data/jax_issue_26021_mass_matrix.hlo
```

This is the **unoptimized native HLO text** (not StableHLO MLIR) extracted from JAX 0.9.1 + MuJoCo MJX 3.5.0 `mass_matrix` function. It's 7349 lines, ~437KB.

## How to Build and Run the Benchmark

### Option A: Using `run_hlo_module` (Recommended)

```bash
# Build the HLO benchmark tool
bazel build -c opt //xla/tools:run_hlo_module

# Run benchmark
bazel-bin/xla/tools/run_hlo_module \
  --input_format=hlo \
  --platform=cpu \
  --reference_platform="" \
  xla/backends/cpu/benchmarks/data/jax_issue_26021_mass_matrix.hlo
```

Note: `run_hlo_module` prints execution time. Use `--reference_platform=""` to skip correctness checking.

### Option B: Using `hlo_runner_main`

```bash
bazel build -c opt //xla/tools:hlo_runner_main

bazel-bin/xla/tools/hlo_runner_main \
  --hlo_file=xla/backends/cpu/benchmarks/data/jax_issue_26021_mass_matrix.hlo \
  --platform=cpu \
  --num_repeats=1000
```

### Option C: Write a custom C++ benchmark

Create a file like `xla/backends/cpu/benchmarks/mass_matrix_benchmark_test.cc`:

```cpp
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla::cpu {

static void BM_MassMatrix(benchmark::State& state) {
  // Read HLO from file
  std::string hlo_path = "xla/backends/cpu/benchmarks/data/jax_issue_26021_mass_matrix.hlo";
  auto hlo_text = tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &hlo_text);

  // Use HloBenchmarkRunner
  HloBenchmarkRunner runner(state);
  runner.Run(hlo_text);
}

BENCHMARK(BM_MassMatrix);

}  // namespace xla::cpu
```

Add corresponding BUILD target. Look at existing benchmarks in `xla/backends/cpu/benchmarks/` for patterns.

### Option D: Using Python xla_client (no bazel needed)

```python
import os, time, statistics
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'true'
# Optionally test with thunks off:
# os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'

import jax
jax.config.update('jax_enable_x64', True)
import numpy as np

# Load StableHLO text (use mass_matrix.hlo from `jax.jit(mjx.mass_matrix).lower(model, data).as_text()`)
with open('/path/to/mass_matrix_stablehlo.hlo', 'r') as f:
    hlo_text = f.read()

from jax._src import xla_bridge
from jax._src.lib import xla_client as xc
client = xla_bridge.get_backend('cpu')
devices = client.devices()
options = xc.CompileOptions()

# Compile (API varies by version)
try:
    compiled = client.compile(hlo_text, compile_options=options)
except:
    compiled = client.compile_and_load(hlo_text, devices[:1], options)

np.random.seed(42)
input_data = np.random.randn(36).astype(np.float64)
input_buf = client.buffer_from_pyval(input_data, devices[0])

# Warmup
for _ in range(50):
    compiled.execute([input_buf])

# Benchmark
N = 3000
times = []
for _ in range(N):
    start = time.perf_counter()
    compiled.execute([input_buf])
    times.append((time.perf_counter() - start) * 1e6)
times.sort()
trimmed = times[100:-100]
median = statistics.median(trimmed)
print(f"Median: {median:.1f}us")
```

## How to Perform the Git Bisect

### For bisect #2 (the "other 3x" with thunks off):

```bash
git bisect start
git bisect good 79fd5733f99b3c0948d7202bc1bbe1ee3980da5c  # 0.4.30 XLA pin
git bisect bad 720b2c53346660e95abbed7cf3309a8b85e979f9   # 0.4.33 XLA pin

# At each step:
# 1. Build: bazel build -c opt //xla/tools:run_hlo_module
# 2. Run with thunks off to isolate non-thunks regressions:
#    XLA_FLAGS=--xla_cpu_use_thunk_runtime=false \
#    bazel-bin/xla/tools/run_hlo_module \
#      --input_format=hlo --platform=cpu --reference_platform="" \
#      xla/backends/cpu/benchmarks/data/jax_issue_26021_mass_matrix.hlo
# 3. If median > 50us → git bisect bad
#    If median < 40us → git bisect good

git bisect run ./bisect_test.sh  # automate with a script
```

### For bisect #4 (0.7.0 → 0.9.1 regression):

First find the XLA commits for JAX 0.7.0 and 0.9.1:
- Look in the JAX repo at `third_party/xla/workspace.bzl` for each release tag
- JAX 0.9.1 XLA pin: `3cc8846c10052cc1c32c4db87866eac4e4cdbccd`

Then bisect between those commits (thunks ON, since that's the only runtime in this range).

## XLA Commit ↔ JAX Version Mapping

| JAX version | XLA commit | Date |
|---|---|---|
| 0.4.30 | `79fd5733f99b3c0948d7202bc1bbe1ee3980da5c` | 2024-06-18 |
| 0.4.31 | `95e3eea8d2aebd55160ed4185a38345ae98ab500` | 2024-07-28 |
| 0.4.33 | `720b2c53346660e95abbed7cf3309a8b85e979f9` | 2024-09-10 |
| 0.9.1  | `3cc8846c10052cc1c32c4db87866eac4e4cdbccd` | 2026-02-26 |

To find others: check out the JAX release tag and read `third_party/xla/workspace.bzl` for the `OPENXLA_XLA_COMMIT` variable.

## Key Files in XLA

- `xla/debug_options_flags.cc` — Where the thunks switch lives (line with `set_xla_cpu_use_thunk_runtime`)
- `xla/backends/cpu/runtime/thunk_executor.cc` — ThunkExecutor implementation
- `xla/backends/cpu/runtime/thunk.h` — Thunk base class
- `xla/backends/cpu/runtime/kernel_thunk.cc` — KernelThunk (runs compiled LLVM kernels)
- `xla/service/cpu/cpu_compiler.cc` — CPU compiler pipeline
- `xla/backends/cpu/benchmarks/` — Existing CPU benchmarks directory

## Benchmark Characteristics

- **Input**: f64[36] (36 generalized coordinates of a humanoid robot)
- **Output**: f64[35,35] (35×35 mass/inertia matrix)
- **HLO ops**: ~7000 operations across ~200 sub-computations
- **Pattern**: Many tiny operations on scalars and small matrices — worst case for thunks dispatch overhead
- **No custom calls**: Pure HLO, no external library dependencies
- **Expected fast time**: ~32us (as seen on 0.4.30)
- **Current slow time**: ~160-200us (5-6x regression on latest XLA)
