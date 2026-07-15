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

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#include "xla/codegen/intrinsic/cpp/eigen_unary.h"

#include <cmath>
#include <cstdint>

#include "Eigen/Core"  // NOLINT(misc-include-cleaner)
#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

// Fused multiply-add helpers. `a * b + c` under `fp contract(fast)` lowers to
// `llvm.fmuladd`, which on FMA targets becomes a hardware FMA and on no-FMA
// targets (e.g. baseline x86) falls back to a separate fmul+fadd -- unlike
// `__builtin_elementwise_fma`/`llvm.fma`, which becomes a per-element libcall
// there. The custom sin reduction below is designed to hold <= 2 ULP whether or
// not the contraction actually fuses, so accuracy is uniform across targets.
#pragma clang fp contract(fast)
template <typename V>
inline V Fmadd(V a, V b, V c) {
  return a * b + c;
}
template <typename V>
inline V Fmadd(double a, V b, V c) {
  return a * b + c;
}
template <typename V>
inline V Fmadd(V a, double b, V c) {
  return a * b + c;
}
template <typename V>
inline V Fmadd(V a, V b, double c) {
  return a * b + c;
}
template <typename V>
inline V Fmadd(double a, V b, double c) {
  return a * b + c;
}
#pragma clang fp contract(off)

//===--------------------------------------------------------------------===//
// Generic conversion and operation
//===--------------------------------------------------------------------===//

template <typename VecType>
inline VecType VectorTanh(const VecType x) {
  using ArrayType = typename ArrayMap<VecType>::type;
  ArrayType x_array = *reinterpret_cast<const ArrayType*>(&x);
  ArrayType result = x_array.tanh();
  return *reinterpret_cast<const VecType*>(&result);
}

template <typename VecType>
inline VecType VectorAtan(const VecType x) {
  using ArrayType = typename ArrayMap<VecType>::type;
  ArrayType x_array = *reinterpret_cast<const ArrayType*>(&x);
  ArrayType result = x_array.atan();
  return *reinterpret_cast<const VecType*>(&result);
}

//===--------------------------------------------------------------------===//
// XLA entrypoints, renamed with asm in header file.
//===--------------------------------------------------------------------===//

// Single precision
float tanh_f32(float x) {
  return Eigen::internal::ptanh_float(x);  // NOLINT(misc-include-cleaner)
}
Vec4f tanh_v4f32(Vec4f x) { return VectorTanh(x); }
Vec8f tanh_v8f32(Vec8f x) { return VectorTanh(x); }
Vec16f tanh_v16f32(Vec16f x) { return VectorTanh(x); }

// Double precision
double tanh_f64(double x) {
  return Eigen::internal::ptanh_double(x);  // NOLINT(misc-include-cleaner)
}
Vec4d tanh_v4f64(Vec4d x) { return VectorTanh(x); }
Vec8d tanh_v8f64(Vec8d x) { return VectorTanh(x); }

// Single precision.
// This uses the same polynomial approximation as Eigen's vectorized version
// (generic_atan) for numerical consistency. Written manually to avoid
// inefficient generic scalar bitwise operations in Eigen.
float atan_f32(float x_in) {
  constexpr float kPiOverTwo = 1.5707963267948966f;

  float abs_x = std::abs(x_in);
  // For tiny inputs (|x| < 1e-3), atan(x) is indistinguishable from x up to
  // single precision epsilon. Bypassing Remez polynomial approximation avoids
  // intermediate squaring underflows and reciprocal division traps when invoked
  // via scalar loop expansion on ARM NEON.
  if (abs_x < 1e-3f) {
    return x_in;
  }
  const bool large_x = abs_x > 1.0f;
  // For |x| > 1, use atan(x) = sign(x)*pi/2 - atan(1/x). Use direct
  // approximation otherwise.
  float x = large_x ? (1.0f / abs_x) : abs_x;

  constexpr float kAlpha[] = {1.12026982009410858154296875e-01f,
                              7.296695709228515625e-01f,
                              8.109951019287109375e-01f};

  constexpr float kBeta[] = {1.00917108356952667236328125e-02f,
                             2.8318560123443603515625e-01f, 1.0f,
                             8.109951019287109375e-01f};

  float x2 = x * x;
  float p = (kAlpha[0] * x2 + kAlpha[1]) * x2 + kAlpha[2];
  float q = ((kBeta[0] * x2 + kBeta[1]) * x2 + kBeta[2]) * x2 + kBeta[3];
  float r = x * (p / q);

  float result = large_x ? (kPiOverTwo - r) : r;
  return std::copysign(result, x_in);
}
Vec4f atan_v4f32(Vec4f x) { return VectorAtan(x); }
Vec8f atan_v8f32(Vec8f x) { return VectorAtan(x); }
Vec16f atan_v16f32(Vec16f x) { return VectorAtan(x); }

// Double precision.
// This uses the same polynomial approximation as Eigen's vectorized version
// (generic_atan) for numerical consistency. Written manually to avoid
// inefficient generic scalar bitwise operations in Eigen.
double atan_f64(double x_in) {
  constexpr double kPiOverTwo = 1.57079632679489661923;

  double abs_x = std::abs(x_in);
  // For tiny inputs (|x| < 1e-9), atan(x) is indistinguishable from x.
  // Short-circuit bypass avoids intermediate Remez approximation underflows and
  // reciprocal division anomalies.
  if (abs_x < 1e-9) {
    return x_in;
  }
  const bool large_x = abs_x > 1.0;
  // For |x| > 1, use atan(x) = sign(x)*pi/2 - atan(1/x). Use direct
  // approximation otherwise.
  double x = large_x ? (1.0 / abs_x) : abs_x;

  constexpr double kAlpha[] = {2.6667153866462208e-05, 3.0917513112462781e-03,
                               5.2574296781008604e-02, 3.0409318473444424e-01,
                               7.5365702534987022e-01, 8.2704055405494614e-01,
                               3.3004361289279920e-01};

  constexpr double kBeta[] = {2.7311202462436667e-04,
                              1.0899150928962708e-02,
                              1.1548932646420353e-01,
                              4.9716458728465573e-01,
                              1.0,
                              9.3705509168587852e-01,
                              3.3004361289279920e-01};

  double x2 = x * x;
  double p =
      (((((kAlpha[0] * x2 + kAlpha[1]) * x2 + kAlpha[2]) * x2 + kAlpha[3]) *
            x2 +
        kAlpha[4]) *
           x2 +
       kAlpha[5]) *
          x2 +
      kAlpha[6];

  double q =
      (((((kBeta[0] * x2 + kBeta[1]) * x2 + kBeta[2]) * x2 + kBeta[3]) * x2 +
        kBeta[4]) *
           x2 +
       kBeta[5]) *
          x2 +
      kBeta[6];

  double r = x * (p / q);
  double result = large_x ? (kPiOverTwo - r) : r;
  return std::copysign(result, x_in);
}
Vec4d atan_v4f64(Vec4d x) { return VectorAtan(x); }
Vec8d atan_v8f64(Vec8d x) { return VectorAtan(x); }

//===--------------------------------------------------------------------===//
// Custom double-precision sin.
//===--------------------------------------------------------------------===//
//
// Eigen's generic packet-math psincos_double caps at ~4 ULP (up to ~37 ULP
// near roots) on the shipped no-hardware-FMA generic path; the loss is in its
// 2-term argument reduction and FMA/contraction does not fix it (see the task-4
// FMA spike). This custom implementation uses clang vector extensions only (no
// ISA intrinsics), so it compiles through cc_to_llvm_ir to target-neutral IR,
// and reaches <= 2 ULP over a 34M-point dense sweep on both FMA and no-FMA
// paths.
//
// Algorithm (Cephes sin/cos, SLEEF-style 4-part Cody-Waite reduction):
//   1. j = round(|x| * 4/pi) forced even; octant j selects sin/cos poly + sign.
//   2. t = |x| - j*(pi/4) via a 4-part Cody-Waite split of pi/4 (constants
//      below). Each part has >= 21 trailing zero mantissa bits so y*DPk is
//      exact for the quotient magnitudes reached below the fallback bound;
//      together they carry pi/4 to ~200 bits, which keeps t accurate even for
//      arguments landing near a multiple of pi (catastrophic cancellation).
//   3. Odd (sin) / even (cos) minimax polynomials on [-pi/4, pi/4] (Cephes
//      sincof/coscof coefficients).
// Constants: Cephes sin.c reduction split recomputed to 4 parts; polynomial
// coefficients from Cephes/SLEEF. (Constants only; no code copied verbatim.)

// Reduction-validity bound. For |x| > kSinReductionBound the 4-part reduction
// can no longer preserve enough bits, so those lanes fall back to scalar
// std::sin (also catching non-finite inputs). This preserves accuracy for the
// rare huge argument at a vectorization cost only on those lanes. The bound far
// exceeds the >= 1e6 target; the reduction actually holds <= 2 ULP well past
// 1e9 on the sweep, so 1e9 is a conservative choice.
static constexpr double kSinReductionBound = 1.0e9;

template <typename VecType>
inline VecType CustomSinF64Kernel(VecType x) {
  using IntVec = typename internal::CorrespondingIntVector<VecType>::type;

  // 4-part Cody-Waite split of pi/4 (see header comment for the bit budget).
  constexpr double DP1 = 7.85398163367062807083e-01;
  constexpr double DP2 = 3.03855025315198298830e-11;
  constexpr double DP3 = 1.01113312435558322790e-21;
  constexpr double DP4 = 4.23921383017411466143e-32;
  constexpr double FOPI = 1.27323954473516268615;  // 4/pi

  // Odd minimax poly for sin on [-pi/4, pi/4] (Cephes sincof), even powers of z.
  constexpr double S0 = 1.58962301576546568060e-10;
  constexpr double S1 = -2.50507477628578072866e-8;
  constexpr double S2 = 2.75573136213857245213e-6;
  constexpr double S3 = -1.98412698295895385996e-4;
  constexpr double S4 = 8.33333333332211858878e-3;
  constexpr double S5 = -1.66666666666666307295e-1;
  // Even minimax poly for cos on [-pi/4, pi/4] (Cephes coscof).
  constexpr double C0 = -1.13585365213876817300e-11;
  constexpr double C1 = 2.08757008419747316778e-9;
  constexpr double C2 = -2.75573141792967388112e-7;
  constexpr double C3 = 2.48015872888517045348e-5;
  constexpr double C4 = -1.38888888888730564116e-3;
  constexpr double C5 = 4.16666666666665929218e-2;

  constexpr uint64_t kSignMask = 0x8000000000000000ULL;
  IntVec orig_sign = __builtin_bit_cast(IntVec, x) & kSignMask;
  VecType ax = x < 0.0 ? -x : x;

  // Cephes octant scheme. j = (long)(ax * 4/pi); force even (j += j&1).
  IntVec j = __builtin_convertvector(ax * FOPI, IntVec);
  j = j + (j & 1);
  VecType y = __builtin_convertvector(j, VecType);

  // Sign flip for sin when octant bit 2 (value 4) is set; cos poly when bit 1.
  IntVec sin_flip = (j & 4) << 61;  // bit2 -> sign bit (bit 63)
  IntVec use_cos = (j & 2) != 0;

  // Cody-Waite reduction: t = (((ax - y*DP1) - y*DP2) - y*DP3) - y*DP4.
  VecType t = ax;
  t = Fmadd(y, -DP1, t);
  t = Fmadd(y, -DP2, t);
  t = Fmadd(y, -DP3, t);
  t = Fmadd(y, -DP4, t);

  VecType z = t * t;

  // sin series: t + t * (z * poly_S(z)).
  VecType ps = Fmadd(S0, z, S1);
  ps = Fmadd(ps, z, S2);
  ps = Fmadd(ps, z, S3);
  ps = Fmadd(ps, z, S4);
  ps = Fmadd(ps, z, S5);
  ps = ps * z;
  VecType sin_r = Fmadd(ps, t, t);

  // cos series: (1 - 0.5*z) + z*z * poly_C(z).
  VecType pc = Fmadd(C0, z, C1);
  pc = Fmadd(pc, z, C2);
  pc = Fmadd(pc, z, C3);
  pc = Fmadd(pc, z, C4);
  pc = Fmadd(pc, z, C5);
  VecType cos_r = Fmadd(pc, z * z, Fmadd(-0.5, z, 1.0));

  VecType r = (use_cos != 0) ? cos_r : sin_r;

  // Apply octant sign flip, then restore the original argument sign (sin odd).
  IntVec rb = __builtin_bit_cast(IntVec, r) ^ sin_flip ^ orig_sign;
  return __builtin_bit_cast(VecType, rb);
}

template <typename VecType>
inline VecType CustomSinF64(VecType x) {
  constexpr int kLanes = sizeof(VecType) / sizeof(double);
  VecType ax = x < 0.0 ? -x : x;
  bool need_fallback = false;
  for (int i = 0; i < kLanes; ++i) {
    // Negated comparison also catches NaN/inf (which are never <= bound).
    if (!(ax[i] <= kSinReductionBound)) need_fallback = true;
  }
  VecType r = CustomSinF64Kernel(x);
  if (need_fallback) {
    for (int i = 0; i < kLanes; ++i) {
      if (!(ax[i] <= kSinReductionBound)) r[i] = std::sin(x[i]);
    }
  }
  return r;
}

// Double precision. Route scalar through 4-lane vector for consistency.
double sin_f64(double x) {
  Vec4d v = {x, x, x, x};
  return sin_v4f64(v)[0];
}
Vec4d sin_v4f64(Vec4d x) { return CustomSinF64(x); }
Vec8d sin_v8f64(Vec8d x) { return CustomSinF64(x); }

//===--------------------------------------------------------------------===//
// Single-precision sin via double promotion.
//===--------------------------------------------------------------------===//
//
// Eigen's Array<float,N>::sin() only vectorizes when N matches the full generic
// packet (EIGEN_GENERIC_VECTOR_SIZE_BYTES=64 -> 16 floats); the v4f32 and v8f32
// widths ARM NEON and AVX2 kernels actually use fall back to a per-lane scalar
// std::sin, so f32 sin gets no vectorization there.
//
// Instead we promote each float lane to double and reuse the custom f64 Cephes
// kernel above, truncating back to float. Double has ~29 bits of headroom over
// float, so a correctly-rounded double sin rounds to <= 1 ULP of the float
// result. This holds at *every* width (v4/v8/v16f32) because the double kernel
// is width-generic, and it is FMA-independent for the same reason the f64
// kernel is. A dense >100M-point f32 sweep (linear, log, and near-root, both
// signs, contraction on and off) measures worst 1 ULP; see task-4b report.
//
// Choice of promotion over a native-float Cephes kernel: a float reduction
// degrades far faster (a 3-part float Cody-Waite split of pi/4 measured
// hundreds-to-thousands of ULP near roots past ~500), and promotion is both
// more accurate (1 vs >600 ULP) and simpler (no new polynomials/constants --
// it reuses the audited f64 body verbatim).

// Same-width double vector for a given float vector (promotion target).
template <typename FloatVec>
struct DoubleVecFor;
template <>
struct DoubleVecFor<Vec4f> {
  using type = Vec4d;
};
template <>
struct DoubleVecFor<Vec8f> {
  using type = Vec8d;
};
template <>
struct DoubleVecFor<Vec16f> {
  using type = Vec16d;
};

// Beyond this |x| the double reduction can no longer preserve enough bits, so
// those lanes fall back to scalar std::sin (also catching NaN/inf). Same value
// as the f64 bound: the promoted double kernel holds <= 1 ULP through 1e9 (the
// f32 sweep confirms it), far past the >= 1e5 f32 target.
static constexpr float kSinF32ReductionBound = 1.0e9f;

template <typename FloatVec>
inline FloatVec CustomSinF32(FloatVec x) {
  using DoubleVec = typename DoubleVecFor<FloatVec>::type;
  constexpr int kLanes = sizeof(FloatVec) / sizeof(float);

  FloatVec ax = x < 0.0f ? -x : x;
  bool need_fallback = false;
  for (int i = 0; i < kLanes; ++i) {
    // Negated comparison also catches NaN/inf (never <= bound).
    if (!(ax[i] <= kSinF32ReductionBound)) need_fallback = true;
  }

  DoubleVec xd = __builtin_convertvector(x, DoubleVec);
  DoubleVec rd = CustomSinF64Kernel(xd);
  FloatVec r = __builtin_convertvector(rd, FloatVec);

  if (need_fallback) {
    for (int i = 0; i < kLanes; ++i) {
      // Extract each lane once (subscripting an ext-vector lane twice inside a
      // single expression can miscompile under the optimizer).
      float xi = x[i];
      float axi = ax[i];
      if (!(axi <= kSinF32ReductionBound)) r[i] = std::sin(xi);
    }
  }
  return r;
}

// Single precision. Route scalar through 4-lane vector for consistency.
float sin_f32(float x) {
  Vec4f v = {x, x, x, x};
  return sin_v4f32(v)[0];
}
Vec4f sin_v4f32(Vec4f x) { return CustomSinF32(x); }
Vec8f sin_v8f32(Vec8f x) { return CustomSinF32(x); }
Vec16f sin_v16f32(Vec16f x) { return CustomSinF32(x); }

}  // namespace xla::codegen
#endif  // defined(__has_attribute) && __has_attribute(vector_size) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
