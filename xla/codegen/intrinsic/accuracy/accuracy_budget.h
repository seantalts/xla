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

// Central registry of accuracy budgets (max allowed ULP error) for each
// intrinsic function and precision level.
//
// When migrating an intrinsic to a new implementation (e.g., Eigen-generated),
// the delta in these budgets is visible in code review. If a migration would
// increase the ULP budget, that must be explicitly approved.
//
// Convention:
//   k<Function><Precision>MaxUlp
//
// Current budgets are based on the existing implementations. Functions not yet
// implemented as intrinsics are given a budget of -1 (no budget yet).

#ifndef XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_BUDGET_H_
#define XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_BUDGET_H_

namespace xla::codegen::intrinsic::accuracy {

// ---------------------------------------------------------------------------
// Existing intrinsics (pinned from current tests)
// ---------------------------------------------------------------------------

// tanh: Rational polynomial approximation (tanh.cc)
// Current impl: 3 ULP F32 (FMA), 3 ULP F64 (no FMA)
// Eigen impl: 5 ULP F32 (from eigen_unary_test.cc)
inline constexpr int kTanhF32MaxUlp = 3;
inline constexpr int kTanhF64MaxUlp = 3;

// exp: Polynomial approximation (exp.cc, F64 only currently)
// polynomial_approximations.cc has F32 version
inline constexpr int kExpF32MaxUlp = 1;
inline constexpr int kExpF64MaxUlp = 1;

// log: Polynomial approximation (polynomial_approximations.cc, F32 only)
inline constexpr int kLogF32MaxUlp = 1;
inline constexpr int kLogF64MaxUlp = 1;  // Not yet implemented as intrinsic

// log1p: Cephes rational polynomial (log1p.cc)
inline constexpr int kLog1pF32MaxUlp = 1;
inline constexpr int kLog1pF64MaxUlp = 1;

// erf: Rational interpolant from Eigen3 (erf.cc)
inline constexpr int kErfF32MaxUlp = 1;
inline constexpr int kErfF64MaxUlp = 1;  // Not yet implemented as intrinsic

// rsqrt: HW intrinsic + Newton-Raphson (rsqrt.cc)
inline constexpr int kRsqrtF32MaxUlp = 1;
inline constexpr int kRsqrtF64MaxUlp = 1;

// ---------------------------------------------------------------------------
// New intrinsics (to be implemented via Eigen)
// Budgets are provisional - set to what Eigen is expected to achieve.
// Will be pinned once implementations exist.
// ---------------------------------------------------------------------------

inline constexpr int kSinF32MaxUlp = 1;
inline constexpr int kSinF64MaxUlp = 1;

inline constexpr int kCosF32MaxUlp = 1;
inline constexpr int kCosF64MaxUlp = 1;

inline constexpr int kTanF32MaxUlp = 1;
inline constexpr int kTanF64MaxUlp = 1;

inline constexpr int kExpm1F32MaxUlp = 1;
inline constexpr int kExpm1F64MaxUlp = 1;

inline constexpr int kSqrtF32MaxUlp = 0;  // Should be exact (HW)
inline constexpr int kSqrtF64MaxUlp = 0;  // Should be exact (HW)

inline constexpr int kAsinF32MaxUlp = 1;
inline constexpr int kAsinF64MaxUlp = 1;

inline constexpr int kAcosF32MaxUlp = 1;
inline constexpr int kAcosF64MaxUlp = 1;

inline constexpr int kAtanF32MaxUlp = 1;
inline constexpr int kAtanF64MaxUlp = 1;

inline constexpr int kSinhF32MaxUlp = 1;
inline constexpr int kSinhF64MaxUlp = 1;

inline constexpr int kCoshF32MaxUlp = 1;
inline constexpr int kCoshF64MaxUlp = 1;

inline constexpr int kAsinhF32MaxUlp = 1;
inline constexpr int kAsinhF64MaxUlp = 1;

inline constexpr int kAcoshF32MaxUlp = 1;
inline constexpr int kAcoshF64MaxUlp = 1;

inline constexpr int kAtanhF32MaxUlp = 1;
inline constexpr int kAtanhF64MaxUlp = 1;

}  // namespace xla::codegen::intrinsic::accuracy

#endif  // XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_BUDGET_H_
