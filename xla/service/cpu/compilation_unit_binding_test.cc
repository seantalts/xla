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

#include "xla/service/cpu/compilation_unit_binding.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

int64_t BufferSizeBytes(const BufferValue& buffer) {
  return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
}

class CompilationUnitBindingTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<BufferAssignment> RunBufferAssignment(HloModule* module) {
    BufferAssigner::Options opts;
    return BufferAssigner::Run(
               module, std::make_unique<DependencyHloOrdering>(module),
               &BufferSizeBytes, &alias_info_,
               [](LogicalBuffer::Color) { return 1; }, std::move(opts))
        .value();
  }

  AliasInfo alias_info_;
};

// The shared callee's private allocations are mapped to the call site's caller
// buffers: each parameter allocation to that operand's caller slice, the result
// allocation to the call's output slice.
TEST_F(CompilationUnitBindingTest, MapsParamsAndResultToCallerSlices) {
  constexpr absl::string_view kHlo = R"(
    HloModule callee
    ENTRY add {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT add = f32[4] add(p0, p1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  std::unique_ptr<BufferAssignment> assignment =
      RunBufferAssignment(module.get());

  // Fabricate distinct caller buffers (recognizable allocation indices).
  BufferAllocation caller_p0(10, 16, LogicalBuffer::Color(0));
  BufferAllocation caller_p1(11, 16, LogicalBuffer::Color(0));
  BufferAllocation caller_res(12, 16, LogicalBuffer::Color(0));
  BufferAllocation::Slice p0_slice(&caller_p0, 0, 16);
  BufferAllocation::Slice p1_slice(&caller_p1, 0, 16);
  BufferAllocation::Slice res_slice(&caller_res, 0, 16);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<BufferAllocation::Slice> binding,
      BuildCalleeBinding(*assignment, {p0_slice, p1_slice}, res_slice));

  // One binding entry per callee private allocation.
  ASSERT_EQ(binding.size(), assignment->Allocations().size());

  for (const BufferAllocation& a : assignment->Allocations()) {
    if (a.is_entry_computation_parameter()) {
      EXPECT_EQ(binding[a.index()],
                a.parameter_number() == 0 ? p0_slice : p1_slice)
          << "param " << a.parameter_number();
    } else if (a.maybe_live_out()) {
      EXPECT_EQ(binding[a.index()], res_slice) << "result";
    }
  }
}

}  // namespace
}  // namespace xla::cpu
