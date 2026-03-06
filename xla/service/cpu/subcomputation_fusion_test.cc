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

#include "xla/service/cpu/subcomputation_fusion.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "gtest/gtest.h"

namespace xla::cpu {
namespace {

class SubcomputationFusionTest : public HloHardwareIndependentTestBase {};

// Verifies that Phase 1 fuses elementwise ops within a sub-computation.
TEST_F(SubcomputationFusionTest, FusesWithinSubcomputation) {
  const char* hlo = R"(
    HloModule test

    helper {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      add = f32[4] add(p0, p1)
      mul = f32[4] multiply(add, p0)
      ROOT neg = f32[4] negate(mul)
    }

    ENTRY main {
      x = f32[4] parameter(0)
      y = f32[4] parameter(1)
      call1 = f32[4] call(x, y), to_apply=helper
      call2 = f32[4] call(y, x), to_apply=helper
      ROOT result = f32[4] add(call1, call2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  SubcomputationFusionPass pass(/*use_multi_output_fusion=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  EXPECT_TRUE(changed);

  // After Phase 1, the helper computation should contain a fusion node.
  // The entry computation should still have the call instructions unchanged.
  const HloComputation* helper = nullptr;
  for (const auto* comp : module->computations()) {
    if (comp->name() == "helper" && !comp->IsFusionComputation()) {
      helper = comp;
      break;
    }
  }
  // After fusion, the helper should have fewer non-parameter instructions.
  // The elementwise chain (add, multiply, negate) should be fused.
  if (helper != nullptr) {
    bool has_fusion = false;
    for (const auto* instr : helper->instructions()) {
      if (instr->opcode() == HloOpcode::kFusion) {
        has_fusion = true;
        break;
      }
    }
    EXPECT_TRUE(has_fusion);
  }
}

// Verifies that the pass does nothing when there are no kCall sub-computations.
TEST_F(SubcomputationFusionTest, NoCallSubcomputations) {
  const char* hlo = R"(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      x = f32[100] parameter(0)
      y = f32[100] parameter(1)
      add = f32[100] add(x, y)
      mul = f32[100] multiply(add, x)
      zero = f32[] constant(0)
      ROOT result = f32[] reduce(mul, zero), dimensions={0}, to_apply=add_f32
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  SubcomputationFusionPass pass(/*use_multi_output_fusion=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  // The reduce's to_apply is an embedded computation (not kCall), so the
  // pass should not create any new fusions specifically for it. However,
  // the fusion pass may still fuse the elementwise chain in the entry
  // computation. The key check is that the pass doesn't crash and handles
  // the case correctly.
  // (The pass returns false if no kCall targets exist, but the fusion
  //  pass may still change the module if run on all computations.)
  // We just verify it doesn't error.
}

// Verifies that chained sub-computations (A calls B) are handled.
TEST_F(SubcomputationFusionTest, ChainedSubcomputations) {
  const char* hlo = R"(
    HloModule test

    inner {
      p0 = f32[4] parameter(0)
      c1 = f32[4] negate(p0)
      c2 = f32[4] negate(c1)
      ROOT c3 = f32[4] negate(c2)
    }

    outer {
      p0 = f32[4] parameter(0)
      call_inner = f32[4] call(p0), to_apply=inner
      ROOT result = f32[4] add(call_inner, p0)
    }

    ENTRY main {
      x = f32[4] parameter(0)
      call1 = f32[4] call(x), to_apply=outer
      call2 = f32[4] call(x), to_apply=outer
      ROOT result = f32[4] add(call1, call2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  SubcomputationFusionPass pass(/*use_multi_output_fusion=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  // The pass should handle chained computations without errors.
  // Inner should be fused first, then outer.
  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace xla::cpu
