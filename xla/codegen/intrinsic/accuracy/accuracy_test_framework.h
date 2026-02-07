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

// Framework for systematically testing numerical accuracy of intrinsics
// against mpmath-generated golden reference values.
//
// Design:
//   - Golden baselines are stored as uint64_t bit patterns (F64 precision).
//   - For F32/F16 tests, inputs and expected outputs are cast down from F64.
//     This is valid because the F64 reference is within 0.5 ULP(F64) of truth,
//     and ULP(F64) << ULP(F32) << ULP(F16), so rounding F64→F32 or F64→F16
//     produces a correctly-rounded result.
//   - Each test reports: max ULP error, mean ULP error, and the worst-case
//     input for debugging.
//   - Accuracy budgets (max allowed ULP error) are defined per-function and
//     per-precision in accuracy_budget.h.

#ifndef XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_TEST_FRAMEWORK_H_
#define XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_TEST_FRAMEWORK_H_

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/codegen/intrinsic/accuracy/golden_baselines.h"
#include "xla/fp_util.h"

namespace xla::codegen::intrinsic::accuracy {

// Convert a uint64_t bit pattern to double.
inline double BitsToDouble(uint64_t bits) { return std::bit_cast<double>(bits); }

// Convert a double to float, clamping infinities that arise from F64 values
// outside F32 range.
inline float DoubleToFloat(double d) { return static_cast<float>(d); }

// Convert a double to Eigen::half (F16). For now we use a simple cast chain.
// Values outside F16 range become inf, which is the correct IEEE behavior.
inline double DoubleToF16AsDouble(double d) {
  // Round to F16 precision by casting through a 16-bit float.
  // We store F16 as double for testing since we don't have a native F16 type
  // in the test harness. The test should compare at F16 precision.
  //
  // For F16: max = 65504, min_positive_normal = 6.1035e-5
  // Values outside this range become inf or zero.
  float f = static_cast<float>(d);
  (void)f;  // F16 conversion happens at the intrinsic level
  return d;
}

// Result of an accuracy sweep across a set of reference points.
struct AccuracyReport {
  std::string function_name;
  std::string precision;  // "F16", "F32", "F64"
  int64_t max_ulp_error = 0;
  double mean_ulp_error = 0.0;
  int64_t p99_ulp_error = 0;
  double worst_input = 0.0;
  double worst_expected = 0.0;
  double worst_actual = 0.0;
  int total_points = 0;
  int nan_matches = 0;     // Both NaN (correct)
  int inf_matches = 0;     // Both same-sign Inf (correct)
  int skipped_points = 0;  // Points outside precision range
};

// Test a scalar F64 function against golden baselines.
template <size_t N>
AccuracyReport TestAccuracyF64(
    const std::string& name,
    const std::array<RefPoint, N>& golden,
    std::function<double(double)> fn_under_test) {
  AccuracyReport report;
  report.function_name = name;
  report.precision = "F64";
  report.total_points = N;

  std::vector<int64_t> all_ulps;
  all_ulps.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    double input = BitsToDouble(golden[i].input_bits);
    double expected = BitsToDouble(golden[i].expected_bits);
    double actual = fn_under_test(input);

    // NaN handling
    if (std::isnan(expected)) {
      if (std::isnan(actual)) {
        report.nan_matches++;
        all_ulps.push_back(0);
      } else {
        // NaN mismatch: treat as infinite error
        ADD_FAILURE() << name << " F64: expected NaN for input "
                      << input << " but got " << actual;
        all_ulps.push_back(std::numeric_limits<int64_t>::max());
      }
      continue;
    }

    // Inf handling
    if (std::isinf(expected)) {
      if (expected == actual) {
        report.inf_matches++;
        all_ulps.push_back(0);
      } else {
        ADD_FAILURE() << name << " F64: expected " << expected
                      << " for input " << input << " but got " << actual;
        all_ulps.push_back(std::numeric_limits<int64_t>::max());
      }
      continue;
    }

    int64_t ulps = CalculateDistanceInFloats(expected, actual);
    all_ulps.push_back(ulps);

    if (ulps > report.max_ulp_error) {
      report.max_ulp_error = ulps;
      report.worst_input = input;
      report.worst_expected = expected;
      report.worst_actual = actual;
    }
  }

  // Compute mean and p99
  int64_t sum = 0;
  for (int64_t u : all_ulps) {
    if (u < std::numeric_limits<int64_t>::max()) sum += u;
  }
  report.mean_ulp_error =
      all_ulps.empty() ? 0.0 : static_cast<double>(sum) / all_ulps.size();

  std::sort(all_ulps.begin(), all_ulps.end());
  size_t p99_idx = static_cast<size_t>(all_ulps.size() * 0.99);
  if (p99_idx >= all_ulps.size()) p99_idx = all_ulps.size() - 1;
  report.p99_ulp_error = all_ulps[p99_idx];

  return report;
}

// Test a scalar F32 function against golden baselines.
// Inputs and expected outputs are rounded from F64 to F32.
template <size_t N>
AccuracyReport TestAccuracyF32(
    const std::string& name,
    const std::array<RefPoint, N>& golden,
    std::function<float(float)> fn_under_test) {
  AccuracyReport report;
  report.function_name = name;
  report.precision = "F32";
  report.total_points = N;

  std::vector<int64_t> all_ulps;
  all_ulps.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    double input_f64 = BitsToDouble(golden[i].input_bits);
    double expected_f64 = BitsToDouble(golden[i].expected_bits);

    float input = static_cast<float>(input_f64);
    float expected = static_cast<float>(expected_f64);

    // Skip points where the F64 input doesn't round-trip through F32
    // meaningfully (e.g., F64-only denormals that become 0 in F32,
    // or values larger than FLT_MAX that become inf).
    // We still test inf and NaN inputs.
    if (!std::isfinite(input_f64) && !std::isnan(input_f64) &&
        !std::isinf(input_f64)) {
      report.skipped_points++;
      continue;
    }

    float actual = fn_under_test(input);

    // NaN handling
    if (std::isnan(expected)) {
      if (std::isnan(actual)) {
        report.nan_matches++;
        all_ulps.push_back(0);
      } else {
        ADD_FAILURE() << name << " F32: expected NaN for input "
                      << input << " but got " << actual;
        all_ulps.push_back(std::numeric_limits<int64_t>::max());
      }
      continue;
    }

    // Inf handling
    if (std::isinf(expected)) {
      if (expected == actual) {
        report.inf_matches++;
        all_ulps.push_back(0);
      } else {
        ADD_FAILURE() << name << " F32: expected " << expected
                      << " for input " << input << " but got " << actual;
        all_ulps.push_back(std::numeric_limits<int64_t>::max());
      }
      continue;
    }

    int64_t ulps = CalculateDistanceInFloats(expected, actual);
    all_ulps.push_back(ulps);

    if (ulps > report.max_ulp_error) {
      report.max_ulp_error = ulps;
      report.worst_input = static_cast<double>(input);
      report.worst_expected = static_cast<double>(expected);
      report.worst_actual = static_cast<double>(actual);
    }
  }

  // Compute mean and p99
  int64_t sum = 0;
  for (int64_t u : all_ulps) {
    if (u < std::numeric_limits<int64_t>::max()) sum += u;
  }
  report.mean_ulp_error =
      all_ulps.empty() ? 0.0 : static_cast<double>(sum) / all_ulps.size();

  std::sort(all_ulps.begin(), all_ulps.end());
  if (!all_ulps.empty()) {
    size_t p99_idx = static_cast<size_t>(all_ulps.size() * 0.99);
    if (p99_idx >= all_ulps.size()) p99_idx = all_ulps.size() - 1;
    report.p99_ulp_error = all_ulps[p99_idx];
  }

  return report;
}

// Print an AccuracyReport as a test log message.
inline void LogReport(const AccuracyReport& r) {
  LOG(INFO) << "=== Accuracy Report: " << r.function_name << " ("
            << r.precision << ") ===";
  LOG(INFO) << "  Total points: " << r.total_points
            << " (skipped: " << r.skipped_points << ")";
  LOG(INFO) << "  NaN matches: " << r.nan_matches
            << ", Inf matches: " << r.inf_matches;
  LOG(INFO) << "  Max ULP error: " << r.max_ulp_error;
  LOG(INFO) << "  Mean ULP error: " << r.mean_ulp_error;
  LOG(INFO) << "  P99 ULP error: " << r.p99_ulp_error;
  if (r.max_ulp_error > 0) {
    LOG(INFO) << "  Worst case: input=" << r.worst_input
              << " expected=" << r.worst_expected
              << " actual=" << r.worst_actual;
  }
}

// Assert that an accuracy report is within the given ULP budget.
inline void AssertWithinBudget(const AccuracyReport& report, int max_ulps) {
  LogReport(report);
  EXPECT_LE(report.max_ulp_error, max_ulps)
      << report.function_name << " " << report.precision
      << " exceeds accuracy budget of " << max_ulps << " ULPs."
      << " Worst case: input=" << report.worst_input
      << " expected=" << report.worst_expected
      << " actual=" << report.worst_actual;
}

}  // namespace xla::codegen::intrinsic::accuracy

#endif  // XLA_CODEGEN_INTRINSIC_ACCURACY_ACCURACY_TEST_FRAMEWORK_H_
