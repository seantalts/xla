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

#include "xla/service/cpu/mega_fusion_pass.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

class MegaFusionPassTest : public HloHardwareIndependentTestBase {};

// A chain of elementwise ops should be fused into a single mega-fusion.
TEST_F(MegaFusionPassTest, ElementwiseChainFused) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      add = f32[100] add(p0, p1)
      neg = f32[100] negate(add)
      ROOT mul = f32[100] multiply(neg, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                           ParseAndReturnVerifiedModule(kHlo));
  MegaFusionPass pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  EXPECT_TRUE(changed);

  // The entry computation root should be a single fusion.
  const HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);

  // The fusion should contain all three ops (add, neg, mul) plus parameters.
  // Fused instruction count = 3 ops + 2 parameters (p0, p1) + 1 root = varies,
  // but we should have exactly one fusion in the computation (plus parameters).
  int fusion_count = 0;
  for (const HloInstruction* instr :
       m->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      fusion_count++;
    }
  }
  EXPECT_EQ(fusion_count, 1);
}

// An instruction with multiple external users should NOT be absorbed.
TEST_F(MegaFusionPassTest, MultiUserNotAbsorbed) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      add = f32[100] add(p0, p1)
      neg = f32[100] negate(add)
      ROOT tuple = (f32[100], f32[100]) tuple(neg, add)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                           ParseAndReturnVerifiedModule(kHlo));
  MegaFusionPass pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  EXPECT_TRUE(changed);

  // 'add' has two users (neg and tuple), so it can't be absorbed into the
  // neg fusion. We should get two separate fusions: one for neg, one for add.
  int fusion_count = 0;
  for (const HloInstruction* instr :
       m->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      fusion_count++;
    }
  }
  EXPECT_EQ(fusion_count, 2);
}

// Non-fusible ops (like dot) should not be fused.
TEST_F(MegaFusionPassTest, DotNotFused) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    ENTRY e {
      p0 = f32[10,20] parameter(0)
      p1 = f32[20,30] parameter(1)
      ROOT dot = f32[10,30] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                           ParseAndReturnVerifiedModule(kHlo));
  MegaFusionPass pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  EXPECT_FALSE(changed);
}

// Mixed fusible/non-fusible: fusible ops get fused, dot stays separate.
TEST_F(MegaFusionPassTest, MixedFusibleAndNonFusible) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    ENTRY e {
      p0 = f32[10,20] parameter(0)
      p1 = f32[20,30] parameter(1)
      dot = f32[10,30] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      p2 = f32[10,30] parameter(2)
      ROOT add = f32[10,30] add(dot, p2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                           ParseAndReturnVerifiedModule(kHlo));
  MegaFusionPass pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  EXPECT_TRUE(changed);

  // The add should be fused, but dot should not be absorbed (not fusible).
  const HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);

  // Dot should still exist as a standalone instruction.
  bool has_dot = false;
  for (const HloInstruction* instr :
       m->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kDot) {
      has_dot = true;
    }
  }
  EXPECT_TRUE(has_dot);
}

// A subsequent run should be a no-op.
TEST_F(MegaFusionPassTest, Idempotent) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      ROOT add = f32[100] add(p0, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                           ParseAndReturnVerifiedModule(kHlo));
  MegaFusionPass pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(changed, pass.Run(m.get()));
  EXPECT_FALSE(changed);
}

// Reduce should be fusible.
TEST_F(MegaFusionPassTest, ReduceFused) {
  static constexpr absl::string_view kHlo = R"(
    HloModule m

    add_computation {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY e {
      p0 = f32[100] parameter(0)
      p1 = f32[100] parameter(1)
      mul = f32[100] multiply(p0, p1)
      zero = f32[] constant(0)
      ROOT reduce = f32[] reduce(mul, zero), dimensions={0},
        to_apply=add_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                           ParseAndReturnVerifiedModule(kHlo));
  MegaFusionPass pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);

  // Both reduce and multiply should be inside the fusion.
  int fusion_count = 0;
  for (const HloInstruction* instr :
       m->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      fusion_count++;
    }
  }
  EXPECT_EQ(fusion_count, 1);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
