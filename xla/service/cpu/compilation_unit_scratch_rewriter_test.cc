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

#include "xla/service/cpu/compilation_unit_scratch_rewriter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

using ::absl_testing::StatusIs;

int64_t BufferSizeBytes(const BufferValue& buffer) {
  return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
}

class CompilationUnitScratchRewriterTest
    : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<BufferAssignment> RunBufferAssignment(HloModule* module) {
    BufferAssigner::Options opts;
    opts.allocate_buffers_for_constants = true;
    return BufferAssigner::Run(
               module, std::make_unique<DependencyHloOrdering>(module),
               &BufferSizeBytes, &alias_info_,
               [](LogicalBuffer::Color) { return 1; }, std::move(opts))
        .value();
  }

  AliasInfo alias_info_;
};

TEST_F(CompilationUnitScratchRewriterTest, RewritesCallSiteToTupleWithScratch) {
  // `a` and `b` are both live when `sub` runs, so at least one is a genuine
  // internal allocation that is neither a parameter nor live-out.
  constexpr absl::string_view kCallee = R"(
    HloModule unit0
    ENTRY e {
      p0 = f32[64] parameter(0)
      p1 = f32[64] parameter(1)
      a = f32[64] add(p0, p1)
      b = f32[64] multiply(p0, p1)
      ROOT sub = f32[64] subtract(a, b)
    })";
  constexpr absl::string_view kParent = R"(
    HloModule parent
    ENTRY e {
      p = f32[64] parameter(0)
      q = f32[64] parameter(1)
      cc = f32[64] custom-call(p, q),
           custom_call_target="_xla_multi_module_call",
           backend_config="unit0",
           api_version=API_VERSION_STATUS_RETURNING_UNIFIED
      ROOT r = f32[64] negate(cc)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto callee,
                          ParseAndReturnUnverifiedModule(kCallee));
  TF_ASSERT_OK_AND_ASSIGN(auto parent,
                          ParseAndReturnUnverifiedModule(kParent));

  std::unique_ptr<BufferAssignment> callee_assignment =
      RunBufferAssignment(callee.get());

  std::vector<const BufferAllocation*> scratch =
      GetCalleeScratchAllocations(*callee_assignment);
  ASSERT_GE(scratch.size(), 1u);

  absl::flat_hash_map<std::string, const BufferAssignment*> assignments{
      {"unit0", callee_assignment.get()}};

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RewriteCompilationUnitScratch(parent.get(), assignments));
  EXPECT_TRUE(changed);

  const HloInstruction* root = parent->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kNegate);
  const HloInstruction* gte = root->operand(0);
  ASSERT_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(gte->tuple_index(), 0);

  const HloInstruction* cc = gte->operand(0);
  ASSERT_EQ(cc->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(cc->custom_call_target(), "_xla_multi_module_call");
  EXPECT_EQ(cc->raw_backend_config_string(), "unit0");

  ASSERT_TRUE(cc->shape().IsTuple());
  ASSERT_EQ(cc->shape().tuple_shapes_size(), 1 + scratch.size());
  for (size_t i = 0; i < scratch.size(); ++i) {
    const Shape& s = cc->shape().tuple_shapes(i + 1);
    EXPECT_EQ(s.element_type(), S8) << "scratch " << i;
    EXPECT_EQ(ShapeUtil::ElementsIn(s), scratch[i]->size()) << "scratch " << i;
  }
}

TEST_F(CompilationUnitScratchRewriterTest, NoScratchNeededLeavesSiteAlone) {
  constexpr absl::string_view kCallee = R"(
    HloModule unit0
    ENTRY e {
      p0 = f32[64] parameter(0)
      ROOT t = f32[64] exponential(p0)
    })";
  constexpr absl::string_view kParent = R"(
    HloModule parent
    ENTRY e {
      p = f32[64] parameter(0)
      cc = f32[64] custom-call(p), custom_call_target="_xla_multi_module_call",
           backend_config="unit0",
           api_version=API_VERSION_STATUS_RETURNING_UNIFIED
      ROOT r = f32[64] negate(cc)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto callee,
                          ParseAndReturnUnverifiedModule(kCallee));
  TF_ASSERT_OK_AND_ASSIGN(auto parent,
                          ParseAndReturnUnverifiedModule(kParent));

  std::unique_ptr<BufferAssignment> callee_assignment =
      RunBufferAssignment(callee.get());
  ASSERT_TRUE(GetCalleeScratchAllocations(*callee_assignment).empty());

  absl::flat_hash_map<std::string, const BufferAssignment*> assignments{
      {"unit0", callee_assignment.get()}};

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, RewriteCompilationUnitScratch(parent.get(), assignments));
  EXPECT_FALSE(changed);

  const HloInstruction* root = parent->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kNegate);
  const HloInstruction* cc = root->operand(0);
  ASSERT_EQ(cc->opcode(), HloOpcode::kCustomCall);
  EXPECT_FALSE(cc->shape().IsTuple());
  EXPECT_TRUE(ShapeUtil::Equal(cc->shape(),
                               ShapeUtil::MakeShape(F32, {64})));
}

TEST_F(CompilationUnitScratchRewriterTest, ErrorsOnUnknownSubmodule) {
  constexpr absl::string_view kParent = R"(
    HloModule parent
    ENTRY e {
      p = f32[64] parameter(0)
      cc = f32[64] custom-call(p), custom_call_target="_xla_multi_module_call",
           backend_config="missing",
           api_version=API_VERSION_STATUS_RETURNING_UNIFIED
      ROOT r = f32[64] negate(cc)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto parent,
                          ParseAndReturnUnverifiedModule(kParent));

  absl::flat_hash_map<std::string, const BufferAssignment*> assignments;

  EXPECT_THAT(RewriteCompilationUnitScratch(parent.get(), assignments),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xla::cpu
