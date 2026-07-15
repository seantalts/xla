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

#include "xla/codegen/intrinsic/cpp/eigen_unary.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_32_ll.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_64_ll.h"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/codegen/intrinsic/test_matchers.h"

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
using ::testing::Not;
using ::xla::codegen::intrinsic::NearUlps;

// Runtime libm sinf, with the argument forced through a volatile so the
// compiler cannot constant-fold it to its (more accurate) internal routine.
// The f32 huge-arg fallback calls std::sin(float) at runtime, i.e. libm sinf;
// a folded std::sin(constant) in a test would round differently (~1 ULP), so
// huge-lane exactness must be checked against this same runtime path.
float SinfRuntime(float x) {
  volatile float v = x;
  return std::sin(static_cast<float>(v));
}

std::string GetFunctionIr(const llvm::Module& module, llvm::StringRef name) {
  llvm::Function* f = module.getFunction(name);
  if (f == nullptr) {
    return "";
  }
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  f->print(stream);
  return ir;
}

constexpr int kTanhUlps = 5;
constexpr int kAtanF32Ulps = 3;
constexpr int kAtanF64Ulps = 2;
// Custom f32 sin promotes each lane to double and reuses the f64 kernel;
// measured worst 1 ULP over a >100M-point dense sweep (contraction on and off).
constexpr int kSinF32Ulps = 1;
// Custom f64 sin: measured worst 2 ULP over a 34M-point dense sweep.
constexpr int kSinF64Ulps = 2;
// Reduction-validity bound in the custom f64 impl; |x| beyond this uses the
// scalar std::sin fallback.
constexpr double kSinF64ReductionBound = 1.0e9;
// Same bound in the custom f32 impl (promotion runs the double kernel, which
// holds through 1e9); |x| beyond this uses the scalar std::sin fallback.
constexpr float kSinF32ReductionBound = 1.0e9f;

TEST(EigenUnaryTest, FastTanhfIsCorrect) {
  Vec16f x = {1.0f,  2.0f,  -1.0f, 4.0f,   8.0f,   16.0f,  32.0f, 200.0f,
              -2.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f, 0.0f,  0.5f};
  Vec16f y = tanh_v16f32(x);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps))
        << x[i] << " " << std::tanh(x[i]);
  }
}

TEST(EigenUnaryTest, FastTanhdIsCorrect) {
  Vec4d x = {1.0, 2.0, -1.0, 4.0};
  Vec4d y = tanh_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

TEST(EigenUnaryTest, LinspaceInputsTanhfCorrectness) {
  constexpr float start = -100.0f;
  constexpr float end = 100.0f;
  constexpr int steps = 1000 * 16;
  constexpr float step = (end - start) / steps;

  for (int i = 0; i < steps; i += 16) {
    Vec16f x;
    for (int j = 0; j < 16; ++j) {
      x[j] = start + (i + j) * step;
    }
    Vec16f y = tanh_v16f32(x);
    for (int j = 0; j < 16; ++j) {
      EXPECT_THAT(y[j], NearUlps(std::tanh(x[j]), kTanhUlps));
    }
  }
}

TEST(EigenUnaryTest, v4f32TanhIsCorrect) {
  Vec4f x = {1.0f, 2.0f, -1.0f, 4.0f};
  Vec4f y = tanh_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

TEST(EigenUnaryTest, v8f32TanhIsCorrect) {
  Vec8f x = {1.0f, 2.0f, -1.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
  Vec8f y = tanh_v8f32(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

TEST(EigenUnaryTest, v8f64TanhIsCorrect) {
  Vec8d x = {1.0, 2.0, -1.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  Vec8d y = tanh_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::tanh(x[i]), kTanhUlps));
  }
}

TEST(EigenUnaryTest, FastTanhfIsVectorized32) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary32LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  bool is_512 = absl::StrContains(ir, "fmul <16 x float>");
  bool is_256 = absl::StrContains(ir, "fmul <8 x float>");
  EXPECT_TRUE(is_512 || is_256);

  std::string v16f32_ir = GetFunctionIr(*module, "xla.unused.tanh.v16f32");
  std::string v8f32_ir = GetFunctionIr(*module, "xla.unused.tanh.v8f32");
  std::string v8f64_ir = GetFunctionIr(*module, "xla.unused.tanh.v8f64");
  std::string v4f64_ir = GetFunctionIr(*module, "xla.unused.tanh.v4f64");

  if (is_512) {
    EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <16 x float>"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <8 x double>"));
    EXPECT_THAT(v16f32_ir, ContainsRegex("<16 x float>.*f0x326F951E"));
  } else {
    EXPECT_THAT(v8f32_ir, ContainsRegex("fmul <8 x float>"));
    EXPECT_THAT(v4f64_ir, ContainsRegex("fmul <4 x double>"));
    EXPECT_THAT(v8f32_ir, ContainsRegex("<8 x float>.*f0x326F951E"));
    EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <8 x float>"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <4 x double>"));
  }
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v8f64"));
}

TEST(EigenUnaryTest, FastTanhfIsVectorized64) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary64LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  std::string v16f32_ir = GetFunctionIr(*module, "xla.unused.tanh.v16f32");
  std::string v8f64_ir = GetFunctionIr(*module, "xla.unused.tanh.v8f64");

  EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <16 x float>"));
  EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <8 x double>"));
  EXPECT_THAT(v16f32_ir, ContainsRegex("<16 x float>.*f0x326F951E"));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.unused.tanh.v8f64"));
}

TEST(EigenUnaryTest, GetCppGenIrStringSelectsCorrectVectorWidth) {
  intrinsics::IntrinsicOptions options_default;
  EXPECT_EQ(GetCppGenIrString(options_default), llvm_ir::kEigenUnary32LlIr);

  options_default.features = "+avx512f";
  EXPECT_EQ(GetCppGenIrString(options_default), llvm_ir::kEigenUnary64LlIr);
}

TEST(EigenUnaryTest, FastAtanfIsCorrect) {
  Vec16f x = {1.0f,  2.0f,  -1.0f, 4.0f,   8.0f,   16.0f,  32.0f, 200.0f,
              -2.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f, 0.0f,  0.5f};
  Vec16f y = atan_v16f32(x);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF32Ulps))
        << x[i] << " " << std::atan(x[i]);
  }
}

TEST(EigenUnaryTest, FastAtandIsCorrect) {
  Vec4d x = {1.0, 2.0, -1.0, 4.0};
  Vec4d y = atan_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF64Ulps));
  }
}

TEST(EigenUnaryTest, LinspaceInputsAtanfCorrectness) {
  constexpr float start = -100.0f;
  constexpr float end = 100.0f;
  constexpr int steps = 1000 * 16;
  constexpr float step = (end - start) / steps;

  for (int i = 0; i < steps; i += 16) {
    Vec16f x;
    for (int j = 0; j < 16; ++j) {
      x[j] = start + (i + j) * step;
    }
    Vec16f y = atan_v16f32(x);
    for (int j = 0; j < 16; ++j) {
      EXPECT_THAT(y[j], NearUlps(std::atan(x[j]), kAtanF32Ulps));
    }
  }
}

TEST(EigenUnaryTest, v4f32AtanIsCorrect) {
  Vec4f x = {1.0f, 2.0f, -1.0f, 4.0f};
  Vec4f y = atan_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF32Ulps));
  }
}

TEST(EigenUnaryTest, v8f32AtanIsCorrect) {
  Vec8f x = {1.0f, 2.0f, -1.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
  Vec8f y = atan_v8f32(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF32Ulps));
  }
}

TEST(EigenUnaryTest, v8f64AtanIsCorrect) {
  Vec8d x = {1.0, 2.0, -1.0, 4.0, 8.0, 16.0, 32.0, 64.0};
  Vec8d y = atan_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::atan(x[i]), kAtanF64Ulps));
  }
}

TEST(EigenUnaryTest, AtanIsVectorized32) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary32LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  bool is_512 = absl::StrContains(ir, "fmul <16 x float>");
  bool is_256 = absl::StrContains(ir, "fmul <8 x float>");
  EXPECT_TRUE(is_512 || is_256);

  std::string v16f32_ir = GetFunctionIr(*module, "xla.atan.v16f32");
  std::string v8f32_ir = GetFunctionIr(*module, "xla.atan.v8f32");
  std::string v8f64_ir = GetFunctionIr(*module, "xla.atan.v8f64");
  std::string v4f64_ir = GetFunctionIr(*module, "xla.atan.v4f64");

  if (is_512) {
    EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <16 x float>"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <8 x double>"));
    EXPECT_THAT(v16f32_ir, ContainsRegex("<16 x float>.*f0x3DE56E67"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("<8 x double>.*f0x3EFBF668DC1807E8"));
  } else {
    EXPECT_THAT(v8f32_ir, ContainsRegex("fmul <8 x float>"));
    EXPECT_THAT(v4f64_ir, ContainsRegex("fmul <4 x double>"));
    EXPECT_THAT(v8f32_ir, ContainsRegex("<8 x float>.*f0x3DE56E67"));
    EXPECT_THAT(v4f64_ir, ContainsRegex("<4 x double>.*f0x3EFBF668DC1807E8"));
    EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <8 x float>"));
    EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <4 x double>"));
  }
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.v8f64"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.f64"));
}

TEST(EigenUnaryTest, AtanIsVectorized64) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary64LlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  std::string v16f32_ir = GetFunctionIr(*module, "xla.atan.v16f32");
  std::string v8f64_ir = GetFunctionIr(*module, "xla.atan.v8f64");

  EXPECT_THAT(v16f32_ir, ContainsRegex("fmul <16 x float>"));
  EXPECT_THAT(v8f64_ir, ContainsRegex("fmul <8 x double>"));
  EXPECT_THAT(v16f32_ir, ContainsRegex("<16 x float>.*f0x3DE56E67"));
  EXPECT_THAT(v8f64_ir, ContainsRegex("<8 x double>.*f0x3EFBF668DC1807E8"));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.x86")));
  EXPECT_THAT(ir, Not(ContainsRegex("llvm.aarch64")));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.v16f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.v8f64"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.f32"));
  EXPECT_THAT(ir, ContainsRegex("xla.atan.f64"));
}

TEST(EigenUnaryTest, AtanEdgeCases) {
  constexpr float kPiOver2f = 1.57079632679489661923f;
  constexpr double kPiOver2 = 1.57079632679489661923;

  // Float NaN
  float nan_f = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(std::isnan(atan_f32(nan_f)));

  // Float Infinity
  float inf_f = std::numeric_limits<float>::infinity();
  EXPECT_NEAR(atan_f32(inf_f), kPiOver2f, 1e-6f);
  EXPECT_NEAR(atan_f32(-inf_f), -kPiOver2f, 1e-6f);

  // Double NaN
  double nan_d = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(atan_f64(nan_d)));

  // Double Infinity
  double inf_d = std::numeric_limits<double>::infinity();
  EXPECT_NEAR(atan_f64(inf_d), kPiOver2, 1e-14);
  EXPECT_NEAR(atan_f64(-inf_d), -kPiOver2, 1e-14);
}

TEST(EigenUnaryTest, SinV4F32) {
  Vec4f x = {0.0f, 0.5f, -1.5f, 3.0f};
  Vec4f y = sin_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF32Ulps)) << x[i];
  }
}

TEST(EigenUnaryTest, SinScalarF32MatchesVector) {
  // Scalar and vector paths must agree so vector-body and scalar-tail loops
  // of one kernel produce identical values.
  for (float v : {0.0f, 1e-4f, 0.5f, -2.5f, 3.14159f, 87.0f}) {
    Vec4f x = {v, v, v, v};
    EXPECT_EQ(sin_f32(v), sin_v4f32(x)[0]) << v;
  }
}

TEST(EigenUnaryTest, SinScalarF64MatchesVector) {
  // Scalar sin_f64 is implemented as sin_v4f64({v,v,v,v})[0]; verify the
  // two are bit-exact across a range that covers fast-path, near-pi, and the
  // fallback region (v > kSinF64ReductionBound).
  for (double v : {0.0, 1e-4, 0.5, -2.5, 3.14159265358979, -1.5707963267948966,
                   1e9 + 1.0, 1e15}) {
    Vec4d x = {v, v, v, v};
    EXPECT_EQ(sin_f64(v), sin_v4f64(x)[0]) << v;
  }
}

TEST(EigenUnaryTest, SinV4F64) {
  Vec4d x = {0.0, 0.5, -1.5, 3.0};
  Vec4d y = sin_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF64Ulps));
  }
}

TEST(EigenUnaryTest, SinSubnormalF32) {
  Vec4f x = {1e-40f, -1e-40f, 1e-45f, 0.0f};
  Vec4f y = sin_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    // sin(x) ~= x for tiny x; FTZ may flush — allow exact or flushed-to-zero.
    EXPECT_TRUE(y[i] == x[i] || y[i] == 0.0f) << x[i] << " -> " << y[i];
  }
}

TEST(EigenUnaryTest, SinV8F32) {
  Vec8f x = {0.0f, 0.5f, -1.5f, 3.0f, 0.0f, 0.5f, -1.5f, 3.0f};
  Vec8f y = sin_v8f32(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF32Ulps)) << x[i];
  }
}

TEST(EigenUnaryTest, SinV16F32) {
  Vec16f x = {0.0f, 0.5f, -1.5f, 3.0f, 0.0f, 0.5f, -1.5f, 3.0f,
              0.0f, 0.5f, -1.5f, 3.0f, 0.0f, 0.5f, -1.5f, 3.0f};
  Vec16f y = sin_v16f32(x);
  for (int i = 0; i < 16; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF32Ulps)) << x[i];
  }
}

// Values near multiples of pi are the worst case for argument reduction
// (catastrophic cancellation as sin(x) -> 0); the others exercise the fast path.
TEST(EigenUnaryTest, SinF32NearPiMultiples) {
  Vec4f x = {3.14159265f, 6.28318531f, 100.0f, 100000.0f};
  Vec4f y = sin_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF32Ulps)) << x[i];
  }
}

// One value in each of the four quadrants, both signs, to exercise octant
// sign/poly selection in the promoted double kernel.
TEST(EigenUnaryTest, SinF32Quadrants) {
  Vec8f x = {0.4f, 2.0f, 4.0f, 5.5f, -0.4f, -2.0f, -4.0f, -5.5f};
  Vec8f y = sin_v8f32(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF32Ulps)) << x[i];
  }
}

// Arguments beyond the reduction bound take the scalar std::sin fallback and
// must match it exactly.
TEST(EigenUnaryTest, SinF32HugeArgFallback) {
  const float b = kSinF32ReductionBound;
  Vec4f x = {b * 10.0f, -b * 100.0f, 1e15f, -1e18f};
  Vec4f y = sin_v4f32(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(y[i], SinfRuntime(x[i])) << x[i];
  }
}

// Mixed small/huge lanes in one vector exercise the per-lane fallback branch
// with both paths simultaneously active: small lanes go through the promoted
// polynomial path (within kSinF32Ulps), huge lanes take the scalar fallback
// (exact match with std::sin).
TEST(EigenUnaryTest, SinF32MixedHugeSmallLanes) {
  Vec4f x = {0.5f, 1e15f, -1.5f, -1e18f};
  Vec4f y = sin_v4f32(x);
  EXPECT_THAT(y[0], NearUlps(std::sin(x[0]), kSinF32Ulps)) << x[0];
  EXPECT_THAT(y[2], NearUlps(std::sin(x[2]), kSinF32Ulps)) << x[2];
  EXPECT_EQ(y[1], SinfRuntime(x[1])) << x[1];
  EXPECT_EQ(y[3], SinfRuntime(x[3])) << x[3];
}

TEST(EigenUnaryTest, SinF32EdgeCases) {
  float nan_f = std::numeric_limits<float>::quiet_NaN();
  float inf_f = std::numeric_limits<float>::infinity();
  Vec4f x = {nan_f, inf_f, -inf_f, -0.0f};
  Vec4f y = sin_v4f32(x);
  EXPECT_TRUE(std::isnan(y[0]));  // sin(NaN)
  EXPECT_TRUE(std::isnan(y[1]));  // sin(+inf)
  EXPECT_TRUE(std::isnan(y[2]));  // sin(-inf)
  // sin(-0.0) == -0.0, sign preserved.
  EXPECT_EQ(y[3], 0.0f);
  EXPECT_TRUE(std::signbit(y[3]));
}

// Reduced-count dense sweep for CI (full >100M-point sweep lives in the audit
// report). Seeded; covers linear + log magnitudes + near-pi multiples over both
// signs, up to the reduction bound. Asserts the <= 1 ULP contract.
TEST(EigenUnaryTest, SinF32DenseSweep) {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<float> lin(-1000.0f, 1000.0f);
  std::uniform_real_distribution<float> mid(-kSinF32ReductionBound,
                                            kSinF32ReductionBound);
  std::uniform_real_distribution<float> lg(-30.0f, 3.0f);
  std::uniform_real_distribution<float> sg(-1.0f, 1.0f);
  std::uniform_real_distribution<float> eps(-1e-3f, 1e-3f);
  std::uniform_int_distribution<int> ik(-300000000, 300000000);

  constexpr int kBatches = 250000;  // 1M points.
  uint32_t worst = 0;
  float worst_x = 0.0f;
  float worst_got = 0.0f;
  float worst_ref = 0.0f;
  for (int b = 0; b < kBatches; ++b) {
    Vec4f x;
    x[0] = lin(rng);
    x[1] = std::pow(10.0f, lg(rng)) * (sg(rng) < 0 ? -1.0f : 1.0f);
    x[2] = mid(rng);
    x[3] = ik(rng) * static_cast<float>(M_PI) + eps(rng);  // near a pi multiple
    Vec4f y = sin_v4f32(x);
    for (int i = 0; i < 4; ++i) {
      float yi = y[i];
      float ref = static_cast<float>(std::sin(static_cast<long double>(x[i])));
      uint32_t ua = __builtin_bit_cast(uint32_t, yi);
      uint32_t ub = __builtin_bit_cast(uint32_t, ref);
      if (static_cast<int32_t>(ua) < 0) ua = 0x80000000u - ua;
      if (static_cast<int32_t>(ub) < 0) ub = 0x80000000u - ub;
      uint32_t u = ua > ub ? ua - ub : ub - ua;
      if (u > worst) {
        worst = u;
        worst_x = x[i];
        worst_got = yi;
        worst_ref = ref;
      }
    }
  }
  EXPECT_LE(worst, static_cast<uint32_t>(kSinF32Ulps))
      << "worst " << worst << " ULP at x=" << worst_x << " got=" << worst_got
      << " ref=" << worst_ref;
}

TEST(EigenUnaryTest, SinV8F64) {
  Vec8d x = {0.0, 0.5, -1.5, 3.0, 0.0, 0.5, -1.5, 3.0};
  Vec8d y = sin_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF64Ulps));
  }
}

// Two values are near pi multiples (worst case for Cody-Waite argument
// reduction -- catastrophic cancellation as sin(x) -> 0); the other two are
// arbitrary in-range points that also exercise the fast path.
TEST(EigenUnaryTest, SinF64NearPiMultiples) {
  Vec4d x = {3.14159265358979, 6.283185307179586, 100.0, 1000000.0};
  Vec4d y = sin_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF64Ulps)) << x[i];
  }
}

// One value in each of the four quadrants, both signs, to exercise octant
// sign/poly selection.
TEST(EigenUnaryTest, SinF64Quadrants) {
  Vec8d x = {0.4, 2.0, 4.0, 5.5, -0.4, -2.0, -4.0, -5.5};
  Vec8d y = sin_v8f64(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_THAT(y[i], NearUlps(std::sin(x[i]), kSinF64Ulps)) << x[i];
  }
}

// Arguments beyond the reduction bound take the scalar std::sin fallback and
// must match it exactly.
TEST(EigenUnaryTest, SinF64HugeArgFallback) {
  const double b = kSinF64ReductionBound;
  Vec4d x = {b * 10.0, -b * 100.0, 1e15, -1e18};
  Vec4d y = sin_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(y[i], std::sin(x[i])) << x[i];
  }
}

// Mixed small/huge lanes in one vector exercise the per-lane fallback branch
// with both paths simultaneously active: small lanes go through the fast
// polynomial path (within kSinF64Ulps), huge lanes take the scalar fallback
// (exact match with std::sin).
TEST(EigenUnaryTest, SinF64MixedHugeSmallLanes) {
  Vec4d x = {0.5, 1e15, -1.5, -1e18};
  Vec4d y = sin_v4f64(x);
  // Small lanes: within ULP budget of std::sin.
  EXPECT_THAT(y[0], NearUlps(std::sin(x[0]), kSinF64Ulps)) << x[0];
  EXPECT_THAT(y[2], NearUlps(std::sin(x[2]), kSinF64Ulps)) << x[2];
  // Huge lanes: exact match with std::sin (scalar fallback).
  EXPECT_EQ(y[1], std::sin(x[1])) << x[1];
  EXPECT_EQ(y[3], std::sin(x[3])) << x[3];
}

TEST(EigenUnaryTest, SinSubnormalF64) {
  Vec4d x = {1e-320, -1e-320, std::numeric_limits<double>::denorm_min(), 0.0};
  Vec4d y = sin_v4f64(x);
  for (int i = 0; i < 4; ++i) {
    // sin(x) ~= x for tiny x; FTZ may flush -- allow exact or flushed-to-zero.
    EXPECT_TRUE(y[i] == x[i] || y[i] == 0.0) << x[i] << " -> " << y[i];
  }
}

TEST(EigenUnaryTest, SinF64EdgeCases) {
  // NaN -> NaN.
  double nan_d = std::numeric_limits<double>::quiet_NaN();
  double inf_d = std::numeric_limits<double>::infinity();
  Vec4d x = {nan_d, inf_d, -inf_d, -0.0};
  Vec4d y = sin_v4f64(x);
  EXPECT_TRUE(std::isnan(y[0]));  // sin(NaN)
  EXPECT_TRUE(std::isnan(y[1]));  // sin(+inf)
  EXPECT_TRUE(std::isnan(y[2]));  // sin(-inf)
  // sin(-0.0) == -0.0, sign preserved.
  EXPECT_EQ(y[3], 0.0);
  EXPECT_TRUE(std::signbit(y[3]));
}

// Reduced-count dense sweep for CI (full 17M-point sweep lives in the audit
// report). Seeded; covers linear + log magnitudes + near-pi multiples over
// both signs, up to the reduction bound. Asserts the <= 2 ULP contract.
TEST(EigenUnaryTest, SinF64DenseSweep) {
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> lin(-1000.0, 1000.0);
  std::uniform_real_distribution<double> mid(-kSinF64ReductionBound,
                                             kSinF64ReductionBound);
  std::uniform_real_distribution<double> lg(-30.0, 3.0);
  std::uniform_real_distribution<double> sg(-1.0, 1.0);
  std::uniform_real_distribution<double> eps(-1e-6, 1e-6);
  std::uniform_int_distribution<long> ik(-300000000L, 300000000L);

  constexpr int kBatches = 250000;  // 1M points.
  uint64_t worst = 0;
  double worst_x = 0.0;
  double worst_got = 0.0;
  double worst_ref = 0.0;
  for (int b = 0; b < kBatches; ++b) {
    Vec4d x;
    x[0] = lin(rng);
    x[1] = std::pow(10.0, lg(rng)) * (sg(rng) < 0 ? -1.0 : 1.0);
    x[2] = mid(rng);
    x[3] = ik(rng) * M_PI + eps(rng);  // near a multiple of pi
    Vec4d y = sin_v4f64(x);
    for (int i = 0; i < 4; ++i) {
      double yi = y[i];
      double ref = static_cast<double>(std::sin(static_cast<long double>(x[i])));
      uint64_t ua = __builtin_bit_cast(uint64_t, yi);
      uint64_t ub = __builtin_bit_cast(uint64_t, ref);
      if (static_cast<int64_t>(ua) < 0) ua = 0x8000000000000000ULL - ua;
      if (static_cast<int64_t>(ub) < 0) ub = 0x8000000000000000ULL - ub;
      uint64_t u = ua > ub ? ua - ub : ub - ua;
      if (u > worst) {
        worst = u;
        worst_x = x[i];
        worst_got = yi;
        worst_ref = ref;
      }
    }
  }
  EXPECT_LE(worst, static_cast<uint64_t>(kSinF64Ulps))
      << "worst " << worst << " ULP at x=" << worst_x << " got=" << worst_got
      << " ref=" << worst_ref;
}

}  // namespace
}  // namespace xla::codegen
