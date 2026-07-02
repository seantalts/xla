/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/cpu/small_scatter_expander.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class SmallScatterExpanderTest : public HloHardwareIndependentTestBase {
 protected:
  // Default 64KB smallness threshold, matching the SmallRegionHoistingPass
  // region byte gate this pass is paired with in cpu_compiler.
  absl::StatusOr<bool> RunPass(HloModule* module,
                               int64_t small_buffer_access_size = 1 << 16) {
    return cpu::SmallScatterExpander(small_buffer_access_size).Run(module);
  }
};

// A small scatter (aggregate operand + result footprint far below the
// threshold) is expanded into a while loop of gather/dynamic-update-slice;
// no scatter instruction survives.
TEST_F(SmallScatterExpanderTest, SmallScatterExpanded) {
  constexpr absl::string_view hlo_string = R"(
    HloModule small_scatter

    add_combiner {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    ENTRY main (operand: f32[16], indices: s32[4,1], updates: f32[4]) -> f32[16] {
      operand = f32[16]{0} parameter(0)
      indices = s32[4,1]{1,0} parameter(1)
      updates = f32[4]{0} parameter(2)
      ROOT sc = f32[16]{0} scatter(operand, indices, updates),
          update_window_dims={}, inserted_window_dims={0},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=add_combiner
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(FindInstruction(m.get(), HloOpcode::kScatter), nullptr);
  EXPECT_NE(FindInstruction(m.get(), HloOpcode::kWhile), nullptr);
}

// A large scatter (800KB operand, 800KB result) stays a scatter so the
// dedicated fusion-emitter scatter kernel keeps handling it.
TEST_F(SmallScatterExpanderTest, LargeScatterUnchanged) {
  constexpr absl::string_view hlo_string = R"(
    HloModule large_scatter

    add_combiner {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    ENTRY main (operand: f32[200000], indices: s32[4,1], updates: f32[4]) -> f32[200000] {
      operand = f32[200000]{0} parameter(0)
      indices = s32[4,1]{1,0} parameter(1)
      updates = f32[4]{0} parameter(2)
      ROOT sc = f32[200000]{0} scatter(operand, indices, updates),
          update_window_dims={}, inserted_window_dims={0},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=add_combiner
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_FALSE(changed);
  EXPECT_NE(FindInstruction(m.get(), HloOpcode::kScatter), nullptr);
}

// The footprint sums ALL operands: here the scatter target is tiny (64B) but
// the indices and updates are ~200KB each, so the scatter must not expand.
TEST_F(SmallScatterExpanderTest, FootprintCountsIndicesAndUpdates) {
  constexpr absl::string_view hlo_string = R"(
    HloModule huge_updates_scatter

    add_combiner {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    ENTRY main (operand: f32[16], indices: s32[50000,1], updates: f32[50000]) -> f32[16] {
      operand = f32[16]{0} parameter(0)
      indices = s32[50000,1]{1,0} parameter(1)
      updates = f32[50000]{0} parameter(2)
      ROOT sc = f32[16]{0} scatter(operand, indices, updates),
          update_window_dims={}, inserted_window_dims={0},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=add_combiner
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_FALSE(changed);
  EXPECT_NE(FindInstruction(m.get(), HloOpcode::kScatter), nullptr);
}

// A small variadic (two-operand) scatter expands: the tuple-shaped result is
// walked per leaf when computing the footprint, and the expansion path
// supports multiple operands.
TEST_F(SmallScatterExpanderTest, VariadicSmallScatterExpanded) {
  constexpr absl::string_view hlo_string = R"(
    HloModule variadic_small_scatter

    add_mul_combiner {
      lhs_a = f32[] parameter(0)
      lhs_b = f32[] parameter(1)
      rhs_a = f32[] parameter(2)
      rhs_b = f32[] parameter(3)
      sum_a = f32[] add(lhs_a, rhs_a)
      prod_b = f32[] multiply(lhs_b, rhs_b)
      ROOT out = (f32[], f32[]) tuple(sum_a, prod_b)
    }

    ENTRY main {
      operand_a = f32[16]{0} parameter(0)
      operand_b = f32[16]{0} parameter(1)
      indices = s32[4,1]{1,0} parameter(2)
      updates_a = f32[4]{0} parameter(3)
      updates_b = f32[4]{0} parameter(4)
      ROOT sc = (f32[16]{0}, f32[16]{0}) scatter(operand_a, operand_b, indices, updates_a, updates_b),
          update_window_dims={}, inserted_window_dims={0},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=add_mul_combiner
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(FindInstruction(m.get(), HloOpcode::kScatter), nullptr);
  EXPECT_NE(FindInstruction(m.get(), HloOpcode::kWhile), nullptr);
}

// Variadic footprint sums across ALL operands and ALL result tuple leaves.
// Each operand is 20KB, so a single operand/result pair (40KB) is below the
// 64KB threshold, but the full sum (2 operands + 2 result leaves = 80KB) is
// above it: the scatter must not expand.
TEST_F(SmallScatterExpanderTest, VariadicScatterFootprintCountsAllOperands) {
  constexpr absl::string_view hlo_string = R"(
    HloModule variadic_large_scatter

    add_mul_combiner {
      lhs_a = f32[] parameter(0)
      lhs_b = f32[] parameter(1)
      rhs_a = f32[] parameter(2)
      rhs_b = f32[] parameter(3)
      sum_a = f32[] add(lhs_a, rhs_a)
      prod_b = f32[] multiply(lhs_b, rhs_b)
      ROOT out = (f32[], f32[]) tuple(sum_a, prod_b)
    }

    ENTRY main {
      operand_a = f32[5000]{0} parameter(0)
      operand_b = f32[5000]{0} parameter(1)
      indices = s32[4,1]{1,0} parameter(2)
      updates_a = f32[4]{0} parameter(3)
      updates_b = f32[4]{0} parameter(4)
      ROOT sc = (f32[5000]{0}, f32[5000]{0}) scatter(operand_a, operand_b, indices, updates_a, updates_b),
          update_window_dims={}, inserted_window_dims={0},
          scatter_dims_to_operand_dims={0}, index_vector_dim=1,
          to_apply=add_mul_combiner
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_FALSE(changed);
  EXPECT_NE(FindInstruction(m.get(), HloOpcode::kScatter), nullptr);
}

}  // namespace
}  // namespace xla
