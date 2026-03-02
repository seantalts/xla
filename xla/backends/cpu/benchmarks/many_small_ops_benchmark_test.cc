/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Benchmark targeting the MJX-like workload pattern from
// https://github.com/jax-ml/jax/issues/26021
//
// The MJX (MuJoCo JAX) robotics workload consists of many small matrix
// operations (kinematics, dynamics computations) on tensors sized for typical
// robots (29-36 DoF). This results in HLO programs with:
//   - Many thunks (50-200+) executed sequentially
//   - Each thunk operating on small buffers (< 512 bytes for scalars/vectors,
//     < 10KB for small matrices)
//   - Mix of kernel thunks (element-wise ops, dots) and custom call thunks
//   - Total execution time is very short (0.02-0.13ms), making per-thunk
//     overhead the dominant factor
//
// This benchmark profiles the overhead of:
//   1. ThunkExecutor sequential dispatch for many small ops
//   2. FFI custom call overhead (call frame pooling, context creation)
//   3. Small dot products and element-wise operations

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

//===----------------------------------------------------------------------===//
// Benchmark 1: Many sequential small ops (MJX-like kinematics pattern)
//
// This simulates the pattern where MJX computes kinematics chains:
// many sequential small matrix multiplications and additions.
//===----------------------------------------------------------------------===//

static void BM_ManySmallSequentialOps(benchmark::State& state,
                                       HloBenchmarkOptions options) {
  int64_t num_ops = state.range(0);
  int64_t n = 6;  // Typical spatial vector size in robotics

  // Build a chain of small additions: each depends on the previous.
  // This forces sequential execution in the thunk executor.
  std::string hlo = "HloModule many_small_ops\n\nENTRY e {\n";
  hlo += absl::StrCat("  p0 = f64[", n, ",", n, "] parameter(0)\n");
  hlo += absl::StrCat("  p1 = f64[", n, ",", n, "] parameter(1)\n");
  hlo += absl::StrCat("  add0 = f64[", n, ",", n, "] add(p0, p1)\n");

  for (int64_t i = 1; i < num_ops; ++i) {
    hlo += absl::StrCat("  add", i, " = f64[", n, ",", n, "] add(add",
                         i - 1, ", p1)\n");
  }

  hlo += absl::StrCat("  ROOT out = f64[", n, ",", n, "] add(add",
                       num_ops - 1, ", p0)\n");
  hlo += "}\n";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F64, {n, n});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Benchmark 2: Many small dot products (MJX mass matrix pattern)
//
// Simulates the pattern where MJX computes mass matrices through
// composite rigid body algorithm: many small matrix multiplications.
//===----------------------------------------------------------------------===//

static void BM_ManySmallDots(benchmark::State& state,
                              HloBenchmarkOptions options) {
  int64_t num_dots = state.range(0);
  int64_t n = 6;  // 6x6 spatial inertia matrices

  std::string hlo = "HloModule many_small_dots\n\nENTRY e {\n";
  hlo += absl::StrCat("  p0 = f64[", n, ",", n, "] parameter(0)\n");
  hlo += absl::StrCat("  p1 = f64[", n, ",", n, "] parameter(1)\n");
  hlo += absl::StrCat("  dot0 = f64[", n, ",", n,
                       "] dot(p0, p1), lhs_contracting_dims={1}, "
                       "rhs_contracting_dims={0}\n");

  for (int64_t i = 1; i < num_dots; ++i) {
    hlo += absl::StrCat("  dot", i, " = f64[", n, ",", n, "] dot(dot",
                         i - 1, ", p1), lhs_contracting_dims={1}, "
                         "rhs_contracting_dims={0}\n");
  }

  hlo += absl::StrCat("  ROOT out = f64[", n, ",", n, "] add(dot",
                       num_dots - 1, ", p0)\n");
  hlo += "}\n";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F64, {n, n});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Benchmark 3: Mixed ops pattern (MJX gravity/RNE pattern)
//
// Simulates the recursive Newton-Euler pattern: mix of dots, adds,
// reshapes, and slices operating on small tensors.
//===----------------------------------------------------------------------===//

static void BM_MixedSmallOps(benchmark::State& state,
                              HloBenchmarkOptions options) {
  int64_t num_joints = state.range(0);

  // Simulates a simplified kinematic chain computation per joint:
  // For each joint: multiply transform, add bias, slice result
  std::string hlo = "HloModule mixed_small_ops\n\nENTRY e {\n";
  hlo += "  p_q = f64[36] parameter(0)\n";       // Joint positions
  hlo += "  p_bias = f64[6,6] parameter(1)\n";   // Bias matrix

  // Initial transform
  hlo += "  slice0 = f64[6] slice(p_q), slice={[0:6]}\n";
  hlo += "  bcast0 = f64[6,6] broadcast(slice0), dimensions={0}\n";
  hlo += "  add0 = f64[6,6] add(bcast0, p_bias)\n";

  for (int64_t i = 1; i < num_joints && i < 6; ++i) {
    int64_t start = i * 6;
    int64_t end = start + 6;
    if (end > 36) break;
    hlo += absl::StrCat("  slice", i, " = f64[6] slice(p_q), slice={[",
                         start, ":", end, "]}\n");
    hlo += absl::StrCat("  bcast", i,
                         " = f64[6,6] broadcast(slice", i,
                         "), dimensions={0}\n");
    hlo += absl::StrCat("  mul", i, " = f64[6,6] dot(add", i - 1,
                         ", bcast", i,
                         "), lhs_contracting_dims={1}, "
                         "rhs_contracting_dims={0}\n");
    hlo += absl::StrCat("  add", i, " = f64[6,6] add(mul", i,
                         ", p_bias)\n");
  }

  int64_t last = std::min(num_joints - 1, int64_t{5});
  hlo += absl::StrCat("  ROOT out = f64[6,6] add(add", last, ", p_bias)\n");
  hlo += "}\n";

  std::minstd_rand0 engine;
  auto q_shape = ShapeUtil::MakeShape(F64, {36});
  auto bias_shape = ShapeUtil::MakeShape(F64, {6, 6});
  auto p_q = *LiteralUtil::CreateRandomLiteral<F64>(q_shape, &engine, 1.0, 0.1);
  auto p_bias =
      *LiteralUtil::CreateRandomLiteral<F64>(bias_shape, &engine, 1.0, 0.1);

  std::vector<const Literal*> args = {&p_q, &p_bias};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Benchmark 4: FFI custom call overhead with varying buffer counts
//
// Tests the overhead of the FFI call path specifically, measuring:
// - ObjectPool call frame retrieval
// - Buffer address resolution
// - ExecutionContext creation
// - FFI handler dispatch
//===----------------------------------------------------------------------===//

static absl::Status NoopFFI2In1Out(
    ffi::Buffer<PrimitiveType::F64> arg0,
    ffi::Buffer<PrimitiveType::F64> arg1,
    ffi::Result<ffi::Buffer<PrimitiveType::F64>> ret0) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kNoopFFI2In1Out, NoopFFI2In1Out,
                       ffi::Ffi::Bind()
                           .Arg<ffi::Buffer<PrimitiveType::F64>>()
                           .Arg<ffi::Buffer<PrimitiveType::F64>>()
                           .Ret<ffi::Buffer<PrimitiveType::F64>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_bm$$noop_2in_1out",
                         "Host", kNoopFFI2In1Out);

static void BM_CustomCallChain(benchmark::State& state,
                                HloBenchmarkOptions options) {
  int64_t num_calls = state.range(0);
  int64_t n = 6;

  // Chain of custom calls, each depending on the previous output.
  // This directly measures per-custom-call overhead.
  std::string hlo = "HloModule custom_call_chain\n\nENTRY e {\n";
  hlo += absl::StrCat("  p0 = f64[", n, ",", n, "] parameter(0)\n");
  hlo += absl::StrCat("  p1 = f64[", n, ",", n, "] parameter(1)\n");
  hlo += absl::StrCat("  cc0 = f64[", n, ",", n,
                       "] custom-call(p0, p1), "
                       "custom_call_target=\"__xla_bm$$noop_2in_1out\", "
                       "api_version=API_VERSION_TYPED_FFI\n");

  for (int64_t i = 1; i < num_calls; ++i) {
    hlo += absl::StrCat("  cc", i, " = f64[", n, ",", n,
                         "] custom-call(cc", i - 1, ", p1), "
                         "custom_call_target=\"__xla_bm$$noop_2in_1out\", "
                         "api_version=API_VERSION_TYPED_FFI\n");
  }

  hlo += absl::StrCat("  ROOT out = f64[", n, ",", n, "] add(cc",
                       num_calls - 1, ", p0)\n");
  hlo += "}\n";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F64, {n, n});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Benchmark 5: Interleaved compute and custom calls
//
// Simulates the real MJX pattern where compute kernels and FFI calls
// are interleaved (as in kinematics + dynamics computations).
//===----------------------------------------------------------------------===//

static void BM_InterleavedComputeAndCustomCalls(benchmark::State& state,
                                                 HloBenchmarkOptions options) {
  int64_t num_iterations = state.range(0);
  int64_t n = 6;

  std::string hlo = "HloModule interleaved\n\nENTRY e {\n";
  hlo += absl::StrCat("  p0 = f64[", n, ",", n, "] parameter(0)\n");
  hlo += absl::StrCat("  p1 = f64[", n, ",", n, "] parameter(1)\n");

  // First iteration
  hlo += absl::StrCat("  dot0 = f64[", n, ",", n,
                       "] dot(p0, p1), lhs_contracting_dims={1}, "
                       "rhs_contracting_dims={0}\n");
  hlo += absl::StrCat("  add0 = f64[", n, ",", n, "] add(dot0, p0)\n");
  hlo += absl::StrCat("  cc0 = f64[", n, ",", n,
                       "] custom-call(add0, p1), "
                       "custom_call_target=\"__xla_bm$$noop_2in_1out\", "
                       "api_version=API_VERSION_TYPED_FFI\n");

  for (int64_t i = 1; i < num_iterations; ++i) {
    hlo += absl::StrCat("  dot", i, " = f64[", n, ",", n, "] dot(cc",
                         i - 1, ", p1), lhs_contracting_dims={1}, "
                         "rhs_contracting_dims={0}\n");
    hlo += absl::StrCat("  add", i, " = f64[", n, ",", n, "] add(dot",
                         i, ", cc", i - 1, ")\n");
    hlo += absl::StrCat("  cc", i, " = f64[", n, ",", n,
                         "] custom-call(add", i, ", p1), "
                         "custom_call_target=\"__xla_bm$$noop_2in_1out\", "
                         "api_version=API_VERSION_TYPED_FFI\n");
  }

  hlo += absl::StrCat("  ROOT out = f64[", n, ",", n, "] add(cc",
                       num_iterations - 1, ", p0)\n");
  hlo += "}\n";

  std::minstd_rand0 engine;
  auto shape = ShapeUtil::MakeShape(F64, {n, n});
  auto p0 = *LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);
  auto p1 = *LiteralUtil::CreateRandomLiteral<F64>(shape, &engine, 1.0, 0.1);

  std::vector<const Literal*> args = {&p0, &p1};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {}, options));
}

//===----------------------------------------------------------------------===//
// Register benchmarks
//===----------------------------------------------------------------------===//

// Many sequential small element-wise ops (tests ThunkExecutor overhead)
XLA_CPU_BENCHMARK(BM_ManySmallSequentialOps)
    ->MeasureProcessCPUTime()
    ->Arg(10)
    ->Arg(25)
    ->Arg(50)
    ->Arg(100)
    ->Arg(200);

// Many small matrix multiplications (tests kernel dispatch overhead)
XLA_CPU_BENCHMARK(BM_ManySmallDots)
    ->MeasureProcessCPUTime()
    ->Arg(5)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50);

// Mixed ops simulating kinematic chain
XLA_CPU_BENCHMARK(BM_MixedSmallOps)
    ->MeasureProcessCPUTime()
    ->Arg(2)
    ->Arg(4)
    ->Arg(6);

// Custom call chain (tests FFI dispatch overhead specifically)
XLA_CPU_BENCHMARK(BM_CustomCallChain)
    ->MeasureProcessCPUTime()
    ->Arg(1)
    ->Arg(5)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50)
    ->Arg(100);

// Interleaved compute + custom calls (tests combined overhead)
XLA_CPU_BENCHMARK(BM_InterleavedComputeAndCustomCalls)
    ->MeasureProcessCPUTime()
    ->Arg(5)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50);

}  // namespace
}  // namespace xla::cpu
