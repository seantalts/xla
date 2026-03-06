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
// transport simulator. This benchmark captures the key computational patterns:
// matrix-vector products (finite difference stencils), element-wise physics
// (source terms, diffusion), reductions (integrals), and a while-loop for
// iterative time stepping.

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

// A torax-like transport simulation step. The HLO models a single implicit
// time-step of a 1D PDE solver on a radial grid of size $n:
//
//  - p0: temperature profile       f32[$n]
//  - p1: density profile            f32[$n]
//  - p2: diffusion coefficients     f32[$n]
//  - p3: source term                f32[$n]
//  - p4: tridiagonal matrix bands   f32[3,$n]  (sub, diag, super)
//  - p5: step count (scalar)        s32[]
//
// Each iteration of the while loop performs:
//   1. Compute transport flux = diffusion * gradient(temperature)
//   2. Apply sources
//   3. Tridiagonal-like matrix-vector product (banded stencil)
//   4. Update temperature via implicit Euler step
//   5. Compute residual norm (reduction) and iterate
//
// The loop runs for a fixed number of Newton iterations (controlled by p5).
static void BM_ToraxTransportStep(benchmark::State& state,
                                  HloBenchmarkOptions options) {
  int64_t n = state.range(0);

  absl::string_view hlo = R"(
    HloModule torax_transport_step_$n

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    max_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT max = f32[] maximum(lhs, rhs)
    }

    // One Newton iteration of the implicit transport solve.
    newton_body {
      param = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      temp = f32[$n] get-tuple-element(param), index=1
      dens = f32[$n] get-tuple-element(param), index=2
      diff = f32[$n] get-tuple-element(param), index=3
      src  = f32[$n] get-tuple-element(param), index=4
      bands = f32[3,$n] get-tuple-element(param), index=5

      // Gradient approximation: diff * (temp[i+1] - temp[i-1]) / 2
      // via slicing and padding to keep shape [$n].
      one = f32[] constant(1)
      zero_scalar = f32[] constant(0)
      two = f32[] constant(2)
      bcast_two = f32[$n] broadcast(two), dimensions={}

      // Shift-left: temp[1:] padded with 0 on the right.
      slice_left = f32[$n_minus_1] slice(temp), slice={[1:$n]}
      pad_left = f32[$n] pad(slice_left, zero_scalar), padding=0_1

      // Shift-right: temp[:-1] padded with 0 on the left.
      slice_right = f32[$n_minus_1] slice(temp), slice={[0:$n_minus_1]}
      pad_right = f32[$n] pad(slice_right, zero_scalar), padding=1_0

      gradient = f32[$n] subtract(pad_left, pad_right)
      gradient_half = f32[$n] divide(gradient, bcast_two)

      // Transport flux = diffusion * gradient.
      flux = f32[$n] multiply(diff, gradient_half)

      // Tridiagonal matrix-vector product: bands[0]*temp_shifted_right +
      //   bands[1]*temp + bands[2]*temp_shifted_left.
      band_sub = f32[1,$n] slice(bands), slice={[0:1], [0:$n]}
      band_sub_r = f32[$n] reshape(band_sub)
      band_diag = f32[1,$n] slice(bands), slice={[1:2], [0:$n]}
      band_diag_r = f32[$n] reshape(band_diag)
      band_sup = f32[1,$n] slice(bands), slice={[2:3], [0:$n]}
      band_sup_r = f32[$n] reshape(band_sup)

      mv_sub = f32[$n] multiply(band_sub_r, pad_right)
      mv_diag = f32[$n] multiply(band_diag_r, temp)
      mv_sup = f32[$n] multiply(band_sup_r, pad_left)
      mv_sum0 = f32[$n] add(mv_sub, mv_diag)
      mv_result = f32[$n] add(mv_sum0, mv_sup)

      // Residual = mv_result - flux - source.
      sub_flux = f32[$n] subtract(mv_result, flux)
      residual = f32[$n] subtract(sub_flux, src)

      // Damped update: temp_new = temp - 0.8 * residual / diag.
      damping = f32[] constant(0.8)
      bcast_damp = f32[$n] broadcast(damping), dimensions={}
      eps = f32[] constant(1e-12)
      bcast_eps = f32[$n] broadcast(eps), dimensions={}
      safe_diag = f32[$n] add(band_diag_r, bcast_eps)
      correction = f32[$n] divide(residual, safe_diag)
      scaled_corr = f32[$n] multiply(bcast_damp, correction)
      temp_new = f32[$n] subtract(temp, scaled_corr)

      // Density update: simple source integration.
      bcast_one = f32[$n] broadcast(one), dimensions={}
      dens_src = f32[$n] multiply(src, bcast_one)
      dens_new = f32[$n] add(dens, dens_src)

      // Increment iteration counter.
      c1 = s32[] constant(1)
      iter_new = s32[] add(iter, c1)

      ROOT result = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n])
        tuple(iter_new, temp_new, dens_new, diff, src, bands)
    }

    newton_cond {
      param = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      max_iter = s32[] constant(5)
      ROOT cmp = pred[] compare(iter, max_iter), direction=LT
    }

    ENTRY main {
      temp0 = f32[$n] parameter(0)
      dens0 = f32[$n] parameter(1)
      diff0 = f32[$n] parameter(2)
      src0  = f32[$n] parameter(3)
      bands0 = f32[3,$n] parameter(4)
      step_count = s32[] parameter(5)

      zero_iter = s32[] constant(0)
      loop_init = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n])
        tuple(zero_iter, temp0, dens0, diff0, src0, bands0)

      loop_result = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n])
        while(loop_init), condition=newton_cond, body=newton_body

      out_temp = f32[$n] get-tuple-element(loop_result), index=1
      out_dens = f32[$n] get-tuple-element(loop_result), index=2

      // Compute integrated quantities (reductions) for diagnostics.
      zero_f32 = f32[] constant(0)
      total_temp = f32[] reduce(out_temp, zero_f32), dimensions={0},
        to_apply=add_f32
      total_dens = f32[] reduce(out_dens, zero_f32), dimensions={0},
        to_apply=add_f32

      neg_inf = f32[] constant(-inf)
      max_temp = f32[] reduce(out_temp, neg_inf), dimensions={0},
        to_apply=max_f32

      ROOT result = (f32[$n], f32[$n], f32[], f32[], f32[])
        tuple(out_temp, out_dens, total_temp, total_dens, max_temp)
    }
  )";

  std::minstd_rand0 engine;

  auto profile_shape = ShapeUtil::MakeShape(F32, {n});
  auto bands_shape = ShapeUtil::MakeShape(F32, {3, n});

  auto temp = *LiteralUtil::CreateRandomLiteral<F32>(profile_shape, &engine,
                                                     10.0f, 1.0f);
  auto dens = *LiteralUtil::CreateRandomLiteral<F32>(profile_shape, &engine,
                                                     1.0f, 0.1f);
  auto diff = *LiteralUtil::CreateRandomLiteral<F32>(profile_shape, &engine,
                                                     0.5f, 0.1f);
  auto src = *LiteralUtil::CreateRandomLiteral<F32>(profile_shape, &engine,
                                                    0.1f, 0.01f);
  auto bands = *LiteralUtil::CreateRandomLiteral<F32>(bands_shape, &engine,
                                                      1.0f, 0.1f);
  auto step = LiteralUtil::CreateR0<int32_t>(1);

  std::vector<const Literal*> args = {&temp, &dens, &diff, &src, &bands, &step};
  CHECK_OK(RunHloBenchmark(
      state, hlo, args,
      {{"$n_minus_1", absl::StrCat(n - 1)},
       {"$n", absl::StrCat(n)}},
      options));
}

static void BM_ToraxTransportStepCompile(benchmark::State& state,
                                         HloBenchmarkOptions options) {
  int64_t n = state.range(0);

  // Same HLO as BM_ToraxTransportStep, but only measures compilation time.
  absl::string_view hlo = R"(
    HloModule torax_transport_step_compile_$n

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    max_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT max = f32[] maximum(lhs, rhs)
    }

    newton_body {
      param = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      temp = f32[$n] get-tuple-element(param), index=1
      dens = f32[$n] get-tuple-element(param), index=2
      diff = f32[$n] get-tuple-element(param), index=3
      src  = f32[$n] get-tuple-element(param), index=4
      bands = f32[3,$n] get-tuple-element(param), index=5

      one = f32[] constant(1)
      zero_scalar = f32[] constant(0)
      two = f32[] constant(2)
      bcast_two = f32[$n] broadcast(two), dimensions={}

      slice_left = f32[$n_minus_1] slice(temp), slice={[1:$n]}
      pad_left = f32[$n] pad(slice_left, zero_scalar), padding=0_1

      slice_right = f32[$n_minus_1] slice(temp), slice={[0:$n_minus_1]}
      pad_right = f32[$n] pad(slice_right, zero_scalar), padding=1_0

      gradient = f32[$n] subtract(pad_left, pad_right)
      gradient_half = f32[$n] divide(gradient, bcast_two)

      flux = f32[$n] multiply(diff, gradient_half)

      band_sub = f32[1,$n] slice(bands), slice={[0:1], [0:$n]}
      band_sub_r = f32[$n] reshape(band_sub)
      band_diag = f32[1,$n] slice(bands), slice={[1:2], [0:$n]}
      band_diag_r = f32[$n] reshape(band_diag)
      band_sup = f32[1,$n] slice(bands), slice={[2:3], [0:$n]}
      band_sup_r = f32[$n] reshape(band_sup)

      mv_sub = f32[$n] multiply(band_sub_r, pad_right)
      mv_diag = f32[$n] multiply(band_diag_r, temp)
      mv_sup = f32[$n] multiply(band_sup_r, pad_left)
      mv_sum0 = f32[$n] add(mv_sub, mv_diag)
      mv_result = f32[$n] add(mv_sum0, mv_sup)

      sub_flux = f32[$n] subtract(mv_result, flux)
      residual = f32[$n] subtract(sub_flux, src)

      damping = f32[] constant(0.8)
      bcast_damp = f32[$n] broadcast(damping), dimensions={}
      eps = f32[] constant(1e-12)
      bcast_eps = f32[$n] broadcast(eps), dimensions={}
      safe_diag = f32[$n] add(band_diag_r, bcast_eps)
      correction = f32[$n] divide(residual, safe_diag)
      scaled_corr = f32[$n] multiply(bcast_damp, correction)
      temp_new = f32[$n] subtract(temp, scaled_corr)

      bcast_one = f32[$n] broadcast(one), dimensions={}
      dens_src = f32[$n] multiply(src, bcast_one)
      dens_new = f32[$n] add(dens, dens_src)

      c1 = s32[] constant(1)
      iter_new = s32[] add(iter, c1)

      ROOT result = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n])
        tuple(iter_new, temp_new, dens_new, diff, src, bands)
    }

    newton_cond {
      param = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      max_iter = s32[] constant(5)
      ROOT cmp = pred[] compare(iter, max_iter), direction=LT
    }

    ENTRY main {
      temp0 = f32[$n] parameter(0)
      dens0 = f32[$n] parameter(1)
      diff0 = f32[$n] parameter(2)
      src0  = f32[$n] parameter(3)
      bands0 = f32[3,$n] parameter(4)
      step_count = s32[] parameter(5)

      zero_iter = s32[] constant(0)
      loop_init = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n])
        tuple(zero_iter, temp0, dens0, diff0, src0, bands0)

      loop_result = (s32[], f32[$n], f32[$n], f32[$n], f32[$n], f32[3,$n])
        while(loop_init), condition=newton_cond, body=newton_body

      out_temp = f32[$n] get-tuple-element(loop_result), index=1
      out_dens = f32[$n] get-tuple-element(loop_result), index=2

      zero_f32 = f32[] constant(0)
      total_temp = f32[] reduce(out_temp, zero_f32), dimensions={0},
        to_apply=add_f32
      total_dens = f32[] reduce(out_dens, zero_f32), dimensions={0},
        to_apply=add_f32

      neg_inf = f32[] constant(-inf)
      max_temp = f32[] reduce(out_temp, neg_inf), dimensions={0},
        to_apply=max_f32

      ROOT result = (f32[$n], f32[$n], f32[], f32[], f32[])
        tuple(out_temp, out_dens, total_temp, total_dens, max_temp)
    }
  )";

  CHECK_OK(CompileHloBenchmark(
      state, hlo,
      {{"$n_minus_1", absl::StrCat(n - 1)},
       {"$n", absl::StrCat(n)}},
      options));
}

// Torax typically uses 25-cell grids for fast runs and up to several hundred
// cells for high-fidelity simulations.
XLA_CPU_BENCHMARK(BM_ToraxTransportStep)
    ->MeasureProcessCPUTime()
    ->Arg(25)
    ->Arg(50)
    ->Arg(100)
    ->Arg(200);

XLA_CPU_BENCHMARK(BM_ToraxTransportStepCompile)
    ->MeasureProcessCPUTime()
    ->Arg(25)
    ->Arg(50)
    ->Arg(100)
    ->Arg(200);

}  // namespace xla::cpu
