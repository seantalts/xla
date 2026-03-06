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

// Benchmark for a torax-like HLO module. Torax is a JAX-based tokamak plasma
// transport simulator (github.com/google-deepmind/torax).
//
// Torax's step function is @jax.jit compiled. Inside the JIT boundary, a
// Newton-Raphson solver uses jax.lax.while_loop. Each Newton iteration
// computes the theta-method residual via dense matrix-vector products:
//
//   residual = lhs_mat @ x_new + lhs_vec - (rhs_mat @ x_old + rhs_vec)
//
// where lhs_mat and rhs_mat are dense [N*C, N*C] matrices (N = grid cells,
// C = number of evolving channels like ion/electron temperature, density,
// current). These matrices are assembled from tridiagonal diffusion/convection
// blocks plus dense source coupling terms. The convergence check computes
// mean(residual^2).
//
// This benchmark models that structure: a while-loop of Newton iterations
// with dense matrix-vector products and a scalar residual reduction.

#include <cstdint>
#include <random>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/backends/cpu/benchmarks/multi_benchmark_config.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Models a single JIT-compiled Torax step with Newton-Raphson solver.
//
// Parameters (matching Torax's theta-method FVM discretization):
//   - x_old:    f32[$m]      state vector from previous timestep
//   - x_guess:  f32[$m]      initial guess for new state
//   - lhs_mat:  f32[$m,$m]   left-hand side matrix (I - dt*theta*D_new)
//   - lhs_vec:  f32[$m]      left-hand side vector
//   - rhs_mat:  f32[$m,$m]   right-hand side matrix
//   - rhs_vec:  f32[$m]      right-hand side vector
//
// where $m = num_cells * num_channels (e.g. 25 cells * 4 channels = 100).
//
// The while-loop runs Newton iterations:
//   1. Compute residual = lhs_mat @ x + lhs_vec - (rhs_mat @ x_old + rhs_vec)
//   2. Compute loss = mean(residual^2)
//   3. Update x -= damping * residual (simplified Newton step)
//   4. Check convergence (iteration count bound, matching lax.while_loop)
static void BM_ToraxTransportStep(benchmark::State& state,
                                  HloBenchmarkOptions options) {
  int64_t m = state.range(0);

  absl::string_view hlo = R"(
    HloModule torax_transport_step_$m

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    // One Newton-Raphson iteration of the implicit theta-method solve.
    newton_body {
      param = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      x = f32[$m] get-tuple-element(param), index=1
      x_old = f32[$m] get-tuple-element(param), index=2
      lhs_mat = f32[$m,$m] get-tuple-element(param), index=3
      lhs_vec = f32[$m] get-tuple-element(param), index=4
      rhs_mat = f32[$m,$m] get-tuple-element(param), index=5
      rhs_vec = f32[$m] get-tuple-element(param), index=6

      // residual = lhs_mat @ x + lhs_vec - (rhs_mat @ x_old + rhs_vec)
      lhs_prod = f32[$m] dot(lhs_mat, x), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      lhs_result = f32[$m] add(lhs_prod, lhs_vec)

      rhs_prod = f32[$m] dot(rhs_mat, x_old), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      rhs_result = f32[$m] add(rhs_prod, rhs_vec)

      residual = f32[$m] subtract(lhs_result, rhs_result)

      // loss = mean(residual^2) - used for convergence check in Torax.
      residual_sq = f32[$m] multiply(residual, residual)
      zero = f32[] constant(0)
      sum_sq = f32[] reduce(residual_sq, zero), dimensions={0}, to_apply=add_f32

      // Simplified Newton update: x -= damping * residual.
      // (Full Torax uses Jacobian-based solve, but the matrix-vector products
      // dominate the computation profile.)
      damping = f32[] constant(0.8)
      bcast_damp = f32[$m] broadcast(damping), dimensions={}
      scaled_residual = f32[$m] multiply(bcast_damp, residual)
      x_new = f32[$m] subtract(x, scaled_residual)

      c1 = s32[] constant(1)
      iter_new = s32[] add(iter, c1)

      ROOT result = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m])
        tuple(iter_new, x_new, x_old, lhs_mat, lhs_vec, rhs_mat, rhs_vec)
    }

    newton_cond {
      param = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      max_iter = s32[] constant(5)
      ROOT cmp = pred[] compare(iter, max_iter), direction=LT
    }

    ENTRY main {
      x_old = f32[$m] parameter(0)
      x_guess = f32[$m] parameter(1)
      lhs_mat = f32[$m,$m] parameter(2)
      lhs_vec = f32[$m] parameter(3)
      rhs_mat = f32[$m,$m] parameter(4)
      rhs_vec = f32[$m] parameter(5)

      zero_iter = s32[] constant(0)
      loop_init = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m])
        tuple(zero_iter, x_guess, x_old, lhs_mat, lhs_vec, rhs_mat, rhs_vec)

      loop_result = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m])
        while(loop_init), condition=newton_cond, body=newton_body

      x_out = f32[$m] get-tuple-element(loop_result), index=1

      // Post-processing: compute integrated quantities for diagnostics.
      zero = f32[] constant(0)
      total = f32[] reduce(x_out, zero), dimensions={0}, to_apply=add_f32

      ROOT result = (f32[$m], f32[]) tuple(x_out, total)
    }
  )";

  std::minstd_rand0 engine;

  auto vec_shape = ShapeUtil::MakeShape(F32, {m});
  auto mat_shape = ShapeUtil::MakeShape(F32, {m, m});

  auto x_old =
      *LiteralUtil::CreateRandomLiteral<F32>(vec_shape, &engine, 1.0f, 0.1f);
  auto x_guess =
      *LiteralUtil::CreateRandomLiteral<F32>(vec_shape, &engine, 1.0f, 0.1f);
  auto lhs_mat =
      *LiteralUtil::CreateRandomLiteral<F32>(mat_shape, &engine, 0.0f, 0.1f);
  auto lhs_vec =
      *LiteralUtil::CreateRandomLiteral<F32>(vec_shape, &engine, 0.0f, 0.01f);
  auto rhs_mat =
      *LiteralUtil::CreateRandomLiteral<F32>(mat_shape, &engine, 0.0f, 0.1f);
  auto rhs_vec =
      *LiteralUtil::CreateRandomLiteral<F32>(vec_shape, &engine, 0.0f, 0.01f);

  std::vector<const Literal*> args = {&x_old,   &x_guess, &lhs_mat,
                                      &lhs_vec, &rhs_mat, &rhs_vec};
  CHECK_OK(RunHloBenchmark(state, hlo, args, {{"$m", absl::StrCat(m)}},
                           options));
}

static void BM_ToraxTransportStepCompile(benchmark::State& state,
                                         HloBenchmarkOptions options) {
  int64_t m = state.range(0);

  // Same HLO as BM_ToraxTransportStep, but only measures compilation time.
  absl::string_view hlo = R"(
    HloModule torax_transport_step_compile_$m

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    newton_body {
      param = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      x = f32[$m] get-tuple-element(param), index=1
      x_old = f32[$m] get-tuple-element(param), index=2
      lhs_mat = f32[$m,$m] get-tuple-element(param), index=3
      lhs_vec = f32[$m] get-tuple-element(param), index=4
      rhs_mat = f32[$m,$m] get-tuple-element(param), index=5
      rhs_vec = f32[$m] get-tuple-element(param), index=6

      lhs_prod = f32[$m] dot(lhs_mat, x), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      lhs_result = f32[$m] add(lhs_prod, lhs_vec)

      rhs_prod = f32[$m] dot(rhs_mat, x_old), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      rhs_result = f32[$m] add(rhs_prod, rhs_vec)

      residual = f32[$m] subtract(lhs_result, rhs_result)

      residual_sq = f32[$m] multiply(residual, residual)
      zero = f32[] constant(0)
      sum_sq = f32[] reduce(residual_sq, zero), dimensions={0}, to_apply=add_f32

      damping = f32[] constant(0.8)
      bcast_damp = f32[$m] broadcast(damping), dimensions={}
      scaled_residual = f32[$m] multiply(bcast_damp, residual)
      x_new = f32[$m] subtract(x, scaled_residual)

      c1 = s32[] constant(1)
      iter_new = s32[] add(iter, c1)

      ROOT result = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m])
        tuple(iter_new, x_new, x_old, lhs_mat, lhs_vec, rhs_mat, rhs_vec)
    }

    newton_cond {
      param = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      max_iter = s32[] constant(5)
      ROOT cmp = pred[] compare(iter, max_iter), direction=LT
    }

    ENTRY main {
      x_old = f32[$m] parameter(0)
      x_guess = f32[$m] parameter(1)
      lhs_mat = f32[$m,$m] parameter(2)
      lhs_vec = f32[$m] parameter(3)
      rhs_mat = f32[$m,$m] parameter(4)
      rhs_vec = f32[$m] parameter(5)

      zero_iter = s32[] constant(0)
      loop_init = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m])
        tuple(zero_iter, x_guess, x_old, lhs_mat, lhs_vec, rhs_mat, rhs_vec)

      loop_result = (s32[], f32[$m], f32[$m], f32[$m,$m], f32[$m], f32[$m,$m], f32[$m])
        while(loop_init), condition=newton_cond, body=newton_body

      x_out = f32[$m] get-tuple-element(loop_result), index=1

      zero = f32[] constant(0)
      total = f32[] reduce(x_out, zero), dimensions={0}, to_apply=add_f32

      ROOT result = (f32[$m], f32[]) tuple(x_out, total)
    }
  )";

  CHECK_OK(
      CompileHloBenchmark(state, hlo, {{"$m", absl::StrCat(m)}}, options));
}

// Torax grid sizes: 25 cells is a fast default, up to ~200 for high fidelity.
// With 4 evolving channels (temp_ion, temp_electron, density, current), the
// state vector dimension $m = num_cells * num_channels.
// Typical dimensions: 25*1=25, 25*4=100, 50*4=200, 100*4=400.
XLA_CPU_BENCHMARK(BM_ToraxTransportStep)
    ->MeasureProcessCPUTime()
    ->Arg(25)
    ->Arg(100)
    ->Arg(200)
    ->Arg(400);

XLA_CPU_BENCHMARK(BM_ToraxTransportStepCompile)
    ->MeasureProcessCPUTime()
    ->Arg(25)
    ->Arg(100)
    ->Arg(200)
    ->Arg(400);

}  // namespace xla::cpu
