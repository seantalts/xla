"""Generate golden baseline reference values for intrinsic accuracy tests.

Computes high-precision reference values using mpmath (arbitrary precision
arithmetic) for all unary math functions that XLA implements as intrinsics.
The output is a C++ header file containing the reference data as constexpr
arrays of doubles. For F32 and F16 tests, the F64 reference values are cast
down at test time, which is correct because the F64 value is within 0.5
ULP(F64) of truth, and ULP(F64) << ULP(F32) << ULP(F16).

Prerequisites:
  pip install mpmath

Usage:
  python xla/codegen/intrinsic/accuracy/generate_golden_baselines.py

This will write:
  xla/codegen/intrinsic/accuracy/golden_baselines.h

To add a new function:
  1. Add an entry to FUNCTIONS below.
  2. Re-run this script.
  3. Rebuild the accuracy tests.
"""

import math
import struct
import sys
from collections import OrderedDict

try:
    import mpmath
except ImportError:
    print("ERROR: mpmath is required. Install with: pip install mpmath",
          file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration: each function has a name, an mpmath callable, a domain
# specification, and the number of sample points (~200 per function).
#
# Domain specification:
#   "linspace": (start, end, count) - evenly spaced
#   "logspace_pos": (start_exp, end_exp, count) - 10^start_exp to 10^end_exp
#   "logspace_neg": same but negated
#   "special": list of specific values
#   "near_zero": (magnitude, count) - values close to zero
#
# We compose these to get ~200 points per function covering:
#   - Primary domain (linspace)
#   - Near-zero behavior (logspace near 0)
#   - Near-boundary behavior (logspace near overflow/underflow)
#   - Special IEEE values (0, -0, inf, -inf, nan, denorm_min, min, max)
# ---------------------------------------------------------------------------

# F64 special values
POS_ZERO = 0.0
NEG_ZERO = -0.0
POS_INF = float('inf')
NEG_INF = float('-inf')
NAN = float('nan')
F64_MIN = 2.2250738585072014e-308  # std::numeric_limits<double>::min()
F64_MAX = 1.7976931348623157e+308  # std::numeric_limits<double>::max()
F64_DENORM_MIN = 5e-324           # std::numeric_limits<double>::denorm_min()
F16_MAX = 65504.0
F32_MAX = 3.4028235e+38

COMMON_SPECIAL = [POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, NAN]

# Values that are exactly representable in F16 (for F16 test coverage)
F16_SPECIAL = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, 0.125, 10.0, -10.0,
               100.0, -100.0, 0.001, -0.001]


def linspace(start, end, count):
    """Generate evenly spaced values."""
    if count == 1:
        return [start]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def logspace_pos(start_exp, end_exp, count):
    """Generate logarithmically spaced positive values."""
    exps = linspace(start_exp, end_exp, count)
    return [10.0 ** e for e in exps]


def logspace_neg(start_exp, end_exp, count):
    """Generate logarithmically spaced negative values."""
    return [-x for x in logspace_pos(start_exp, end_exp, count)]


# ---------------------------------------------------------------------------
# Function definitions
# ---------------------------------------------------------------------------

FUNCTIONS = OrderedDict()


def _define(name, mpmath_fn, samples):
    """Register a function with its mpmath implementation and sample points."""
    FUNCTIONS[name] = {"mpmath_fn": mpmath_fn, "samples": samples}


# tanh: domain is all reals, saturates at ~+-8 for F32
_define("tanh", mpmath.tanh, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-10.0, 10.0, 120) +      # primary domain
    linspace(-100.0, -10.0, 15) +      # saturation region
    linspace(10.0, 100.0, 15) +        # saturation region
    logspace_pos(-38, -1, 15) +         # near zero positive
    logspace_neg(-38, -1, 15) +         # near zero negative
    [F64_MIN, -F64_MIN, F64_MAX, -F64_MAX]
))

# exp: domain is all reals, overflows at ~709 for F64, ~88 for F32
_define("exp", mpmath.exp, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-20.0, 20.0, 100) +        # primary domain
    linspace(-750.0, -20.0, 20) +       # underflow region
    linspace(20.0, 710.0, 20) +         # overflow region
    logspace_pos(-38, -1, 15) +          # near zero positive
    logspace_neg(-38, -1, 15) +          # near zero negative
    [F64_MIN, -F64_MIN, 88.0, -88.0, 709.0, -709.0]
))

# log: domain is (0, +inf)
_define("log", mpmath.log, (
    [POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, NAN] +
    [1.0, 0.5, 2.0, 10.0, 100.0, 0.25, 0.125, 0.001] +
    linspace(0.01, 10.0, 80) +          # primary domain
    linspace(10.0, 1000.0, 30) +        # larger values
    logspace_pos(-308, 308, 40) +        # full range
    logspace_pos(-38, -1, 20) +          # near zero
    [F64_MIN, F64_MAX, F64_DENORM_MIN, -1.0, -F64_MIN]
))

# log1p: domain is (-1, +inf), accurate near zero
_define("log1p", mpmath.log1p, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-0.99, 10.0, 80) +         # primary domain
    linspace(10.0, 1000.0, 20) +        # larger values
    logspace_pos(-38, -1, 30) +          # near zero (where log1p matters)
    logspace_neg(-38, -1, 20) +          # near zero negative
    logspace_pos(0, 308, 20) +           # large values
    [F64_MIN, -F64_MIN, F64_MAX, -1.0 + 1e-15]
))

# erf: domain is all reals, saturates at ~+-3.9 for F32
_define("erf", mpmath.erf, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-6.0, 6.0, 120) +          # primary domain
    linspace(-100.0, -6.0, 10) +        # saturation region
    linspace(6.0, 100.0, 10) +          # saturation region
    logspace_pos(-38, -1, 20) +          # near zero positive
    logspace_neg(-38, -1, 20) +          # near zero negative
    [F64_MIN, -F64_MIN, F64_MAX, -F64_MAX]
))

# rsqrt: domain is (0, +inf)
_define("rsqrt", lambda x: 1 / mpmath.sqrt(x), (
    [POS_ZERO, POS_INF, NEG_INF, NAN] +
    [1.0, 4.0, 0.25, 100.0, 0.01, 2.0, 0.5, 10.0] +
    linspace(0.001, 100.0, 80) +         # primary domain
    logspace_pos(-308, 308, 60) +         # full range
    logspace_pos(-38, -1, 20) +           # near zero
    [F64_MIN, F64_MAX, F64_DENORM_MIN, -1.0, NEG_ZERO]
))

# sin: domain is all reals, periodic
_define("sin", mpmath.sin, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-4 * math.pi, 4 * math.pi, 120) +   # several periods
    logspace_pos(-38, -1, 20) +                     # near zero
    logspace_neg(-38, -1, 20) +                     # near zero negative
    [F64_MIN, -F64_MIN, 1e10, -1e10, 1e15, -1e15]
))

# cos: domain is all reals, periodic
_define("cos", mpmath.cos, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-4 * math.pi, 4 * math.pi, 120) +
    logspace_pos(-38, -1, 20) +
    logspace_neg(-38, -1, 20) +
    [F64_MIN, -F64_MIN, 1e10, -1e10, 1e15, -1e15]
))

# expm1: domain is all reals, accurate near zero
_define("expm1", mpmath.expm1, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-20.0, 20.0, 80) +         # primary domain
    linspace(-750.0, -20.0, 15) +       # underflow region
    linspace(20.0, 710.0, 15) +         # overflow region
    logspace_pos(-38, -1, 30) +          # near zero (where expm1 matters)
    logspace_neg(-38, -1, 30) +          # near zero negative
    [F64_MIN, -F64_MIN]
))

# sqrt: domain is [0, +inf)
_define("sqrt", mpmath.sqrt, (
    [POS_ZERO, NEG_ZERO, POS_INF, NAN] +
    [1.0, 4.0, 9.0, 16.0, 0.25, 0.01, 2.0, 0.5, 100.0] +
    linspace(0.001, 100.0, 80) +
    logspace_pos(-308, 308, 60) +
    logspace_pos(-38, -1, 20) +
    [F64_MIN, F64_MAX, F64_DENORM_MIN, -1.0]
))

# tan: domain is all reals except odd multiples of pi/2
_define("tan", mpmath.tan, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-1.5, 1.5, 80) +           # within one period
    linspace(-3.1, 3.1, 40) +           # near +-pi
    logspace_pos(-38, -1, 20) +
    logspace_neg(-38, -1, 20) +
    [F64_MIN, -F64_MIN]
    # Avoid exact multiples of pi/2 where tan is undefined
))

# asin: domain is [-1, 1]
_define("asin", mpmath.asin, (
    [POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, NAN] +
    linspace(-1.0, 1.0, 120) +
    logspace_pos(-38, -1, 20) +
    logspace_neg(-38, -1, 20) +
    [F64_MIN, -F64_MIN, 1.0, -1.0, 2.0, -2.0]
))

# acos: domain is [-1, 1]
_define("acos", mpmath.acos, (
    [POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, NAN] +
    linspace(-1.0, 1.0, 120) +
    logspace_pos(-38, -1, 20) +
    logspace_neg(-38, -1, 20) +
    [F64_MIN, -F64_MIN, 1.0, -1.0, 2.0, -2.0]
))

# atan: domain is all reals
_define("atan", mpmath.atan, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-100.0, 100.0, 120) +
    logspace_pos(-38, 38, 30) +
    logspace_neg(-38, 38, 30) +
    [F64_MIN, -F64_MIN, F64_MAX, -F64_MAX]
))

# sinh: domain is all reals
_define("sinh", mpmath.sinh, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-10.0, 10.0, 100) +
    linspace(-710.0, -10.0, 15) +
    linspace(10.0, 710.0, 15) +
    logspace_pos(-38, -1, 20) +
    logspace_neg(-38, -1, 20) +
    [F64_MIN, -F64_MIN]
))

# cosh: domain is all reals
_define("cosh", mpmath.cosh, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-10.0, 10.0, 100) +
    linspace(-710.0, -10.0, 15) +
    linspace(10.0, 710.0, 15) +
    logspace_pos(-38, -1, 20) +
    logspace_neg(-38, -1, 20) +
    [F64_MIN, -F64_MIN]
))

# asinh: domain is all reals
_define("asinh", mpmath.asinh, (
    COMMON_SPECIAL +
    F16_SPECIAL +
    linspace(-100.0, 100.0, 100) +
    logspace_pos(-38, 38, 30) +
    logspace_neg(-38, 38, 30) +
    [F64_MIN, -F64_MIN, F64_MAX, -F64_MAX]
))

# acosh: domain is [1, +inf)
_define("acosh", mpmath.acosh, (
    [POS_INF, NEG_INF, NAN, 0.5, -1.0] +  # out-of-domain tests
    linspace(1.0, 100.0, 100) +
    logspace_pos(0, 308, 40) +
    [1.0 + 1e-15, 1.0 + 1e-10, 1.0 + 1e-5, F64_MAX]
))

# atanh: domain is (-1, 1)
_define("atanh", mpmath.atanh, (
    [POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, NAN] +
    linspace(-0.999, 0.999, 120) +
    logspace_pos(-38, -1, 20) +
    logspace_neg(-38, -1, 20) +
    [F64_MIN, -F64_MIN, 1.0, -1.0, 2.0, -2.0]
))


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------

def double_to_hex(val):
    """Convert a Python float to its IEEE 754 hex representation."""
    return '0x{:016X}'.format(struct.unpack('>Q', struct.pack('>d', val))[0])


# Functions where f(-0) = -0 per IEEE 754
_SIGN_PRESERVING_AT_ZERO = {"tanh", "sin", "asin", "atan", "sinh", "asinh",
                            "atanh", "expm1", "log1p", "tan"}


def compute_reference(mpmath_fn, x, fn_name=""):
    """Compute f(x) using mpmath at 50 decimal digits of precision."""
    mpmath.mp.dps = 50
    try:
        if math.isnan(x):
            return float('nan')
        # Handle -0.0 explicitly: mpmath doesn't preserve signed zero
        if x == 0.0 and math.copysign(1.0, x) < 0:
            if fn_name in _SIGN_PRESERVING_AT_ZERO:
                return -0.0
        if math.isinf(x):
            result = mpmath_fn(mpmath.mpf(x))
            if isinstance(result, mpmath.mpc):
                return float('nan')  # Complex result from real input
            return float(result)
        result = mpmath_fn(mpmath.mpf(x))
        if isinstance(result, mpmath.mpc):
            # Function returned complex for real input (e.g., log(-1), sqrt(-1))
            return float('nan')
        # Convert to Python float (F64) - this is correctly rounded to F64
        return float(result)
    except (ValueError, ZeroDivisionError, mpmath.libmp.libhyper.NoConvergence,
            OverflowError):
        return float('nan')


def deduplicate_samples(samples):
    """Remove duplicate inputs, preserving order. Handles -0.0 vs 0.0."""
    seen = set()
    result = []
    for x in samples:
        # Use the hex representation to distinguish -0.0 from 0.0
        key = double_to_hex(x) if not math.isnan(x) else "NAN"
        if key not in seen:
            seen.add(key)
            result.append(x)
    return result


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

HEADER_TEMPLATE = """\
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

// This file is generated by generate_golden_baselines.py. Do not edit!
//
// Each entry is a pair of (input, expected_output) stored as doubles.
// For F32 and F16 tests, cast both input and expected output down to the
// target precision. The F64 reference is accurate enough that this rounding
// produces correctly-rounded F32/F16 values.

#ifndef XLA_CODEGEN_INTRINSIC_ACCURACY_GOLDEN_BASELINES_H_
#define XLA_CODEGEN_INTRINSIC_ACCURACY_GOLDEN_BASELINES_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace xla::codegen::intrinsic::accuracy {{

// A reference data point stored as uint64_t bit patterns for exact
// representation of IEEE 754 doubles (including -0, NaN, Inf).
// Use AsDouble() to convert to double at test time.
struct RefPoint {{
  uint64_t input_bits;
  uint64_t expected_bits;
}};

{function_data}

}}  // namespace xla::codegen::intrinsic::accuracy

#endif  // XLA_CODEGEN_INTRINSIC_ACCURACY_GOLDEN_BASELINES_H_
"""


def generate_function_data(name, info):
    """Generate C++ array for a single function."""
    mpmath_fn = info["mpmath_fn"]
    samples = deduplicate_samples(info["samples"])
    count = len(samples)

    lines = []
    lines.append(f"// {name}: {count} reference points")
    lines.append(
        f"inline constexpr std::array<RefPoint, {count}> "
        f"kGolden{name.title().replace('1P', '1p')} = {{{{"
    )

    for i, x in enumerate(samples):
        y = compute_reference(mpmath_fn, x, fn_name=name)
        # Use hex doubles for exact representation
        x_hex = double_to_hex(x)
        y_hex = double_to_hex(y)

        # Add a human-readable comment
        if math.isnan(x):
            x_str = "NaN"
        elif math.isinf(x):
            x_str = "+Inf" if x > 0 else "-Inf"
        else:
            x_str = f"{x:.6g}"

        if math.isnan(y):
            y_str = "NaN"
        elif math.isinf(y):
            y_str = "+Inf" if y > 0 else "-Inf"
        else:
            y_str = f"{y:.15g}"

        comma = "," if i < count - 1 else ""
        lines.append(
            f"    // {x_str} -> {y_str}"
        )
        lines.append(
            f"    RefPoint{{{x_hex}ULL, {y_hex}ULL}}{comma}"
        )

    lines.append("}};")
    lines.append("")
    return "\n".join(lines)


def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "golden_baselines.h")

    print(f"Generating golden baselines for {len(FUNCTIONS)} functions...")

    function_data_parts = []
    for name, info in FUNCTIONS.items():
        samples = deduplicate_samples(info["samples"])
        print(f"  {name}: {len(samples)} reference points")
        function_data_parts.append(generate_function_data(name, info))

    content = HEADER_TEMPLATE.format(
        function_data="\n".join(function_data_parts)
    )

    with open(output_path, "w") as f:
        f.write(content)

    print(f"\nWrote {output_path}")
    print("Run clang-format on the output if needed:")
    print(f"  clang-format -i -style=Google {output_path}")


if __name__ == "__main__":
    main()
