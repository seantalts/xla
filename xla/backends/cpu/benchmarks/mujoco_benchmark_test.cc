/* Copyright 2025 The OpenXLA Authors.

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

// Benchmark for MuJoCo/MJX mass_matrix HLO (JAX issue #26021).
//
// This HLO module comes from MJX's rigid-body simulation pipeline. It contains
// many small sub-computations (remainder, cross_product, norm, take, etc.)
// called from dozens of sites. Fusion quality depends on the compiler's ability
// to see through these call boundaries and discover producer-consumer
// opportunities.

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/path.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static std::string GetMassMatrixHloPath() {
  return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "backends", "cpu",
                           "benchmarks", "data",
                           "jax_issue_26021_mass_matrix.hlo");
}

// Benchmark end-to-end execution of the MJX mass_matrix computation.
static void BM_MjxMassMatrix(benchmark::State& state,
                              HloBenchmarkOptions options) {
  auto result = LoadHloModuleAndMaybeIterationLiterals(GetMassMatrixHloPath());
  CHECK_OK(result.status());
  auto& [module, iteration_literals] = *result;

  // The module takes f64[36] input and produces f64[35,35] output.
  std::minstd_rand0 engine;
  auto input_shape = ShapeUtil::MakeShape(F64, {36});
  auto input =
      *LiteralUtil::CreateRandomLiteral<F64>(input_shape, &engine, 0.0, 1.0);

  std::vector<const Literal*> args = {&input};
  CHECK_OK(RunHloBenchmark(state, std::move(module), args, options));
}

// Benchmark compilation time for the MJX mass_matrix computation.
static void BM_MjxMassMatrixCompile(benchmark::State& state,
                                     HloBenchmarkOptions options) {
  auto result = LoadHloModuleAndMaybeIterationLiterals(GetMassMatrixHloPath());
  CHECK_OK(result.status());
  auto& [module, iteration_literals] = *result;

  CHECK_OK(CompileHloBenchmark(state, std::move(module), options));
}

XLA_CPU_BENCHMARK(BM_MjxMassMatrix)->MeasureProcessCPUTime();

XLA_CPU_BENCHMARK(BM_MjxMassMatrixCompile)->MeasureProcessCPUTime();

}  // namespace xla::cpu
