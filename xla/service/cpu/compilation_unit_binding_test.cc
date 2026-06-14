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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

int64_t BufferSizeBytes(const BufferValue& buffer) {
  return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
}

class CompilationUnitBindingTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<BufferAssignment> RunBufferAssignment(HloModule* module) {
    BufferAssigner::Options opts;
    // So that callee constants surface as constant allocations (mirroring how
    // the CPU backend assigns buffers for the shared callee).
    opts.allocate_buffers_for_constants = true;
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

  CalleeCallerSlices caller;
  caller.params[{0, ShapeIndex{}}] = p0_slice;
  caller.params[{1, ShapeIndex{}}] = p1_slice;
  caller.results[ShapeIndex{}] = res_slice;

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<BufferAllocation::Slice> binding,
      BuildCalleeBinding(*assignment, *module->entry_computation(), caller));

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

// A callee with internal (scratch) allocations binds each one to the next
// caller scratch slice, in order of increasing callee allocation index.
TEST_F(CompilationUnitBindingTest, MapsInternalAllocationsToScratchSlices) {
  // `a` and `b` are both live when `sub` runs, so at least one is a genuine
  // internal allocation that is neither a parameter nor live-out.
  constexpr absl::string_view kHlo = R"(
    HloModule callee
    ENTRY f {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      a = f32[4] add(p0, p1)
      b = f32[4] multiply(p0, p1)
      ROOT sub = f32[4] subtract(a, b)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  std::unique_ptr<BufferAssignment> assignment =
      RunBufferAssignment(module.get());

  // Collect the internal allocations (neither parameter nor live-out), in
  // ascending index order.
  std::vector<int64_t> internal_indices;
  for (const BufferAllocation& a : assignment->Allocations()) {
    if (!a.is_entry_computation_parameter() && !a.maybe_live_out()) {
      internal_indices.push_back(a.index());
    }
  }
  ASSERT_FALSE(internal_indices.empty()) << "test needs >=1 internal allocation";

  BufferAllocation caller_p0(10, 16, LogicalBuffer::Color(0));
  BufferAllocation caller_p1(11, 16, LogicalBuffer::Color(0));
  BufferAllocation caller_res(12, 16, LogicalBuffer::Color(0));
  BufferAllocation::Slice p0_slice(&caller_p0, 0, 16);
  BufferAllocation::Slice p1_slice(&caller_p1, 0, 16);
  BufferAllocation::Slice res_slice(&caller_res, 0, 16);

  // One fabricated scratch slice per internal allocation (indices 20, 21, ...).
  std::vector<BufferAllocation> scratch_allocs;
  scratch_allocs.reserve(internal_indices.size());
  std::vector<BufferAllocation::Slice> scratch_slices;
  for (size_t i = 0; i < internal_indices.size(); ++i) {
    scratch_allocs.emplace_back(20 + i, 16, LogicalBuffer::Color(0));
  }
  for (BufferAllocation& alloc : scratch_allocs) {
    scratch_slices.emplace_back(&alloc, 0, 16);
  }

  CalleeCallerSlices caller;
  caller.params[{0, ShapeIndex{}}] = p0_slice;
  caller.params[{1, ShapeIndex{}}] = p1_slice;
  caller.results[ShapeIndex{}] = res_slice;
  caller.scratch = scratch_slices;

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<BufferAllocation::Slice> binding,
      BuildCalleeBinding(*assignment, *module->entry_computation(), caller));

  ASSERT_EQ(binding.size(), assignment->Allocations().size());
  // Internal allocations consume scratch slices in ascending index order.
  for (size_t k = 0; k < internal_indices.size(); ++k) {
    EXPECT_EQ(binding[internal_indices[k]], scratch_slices[k])
        << "internal #" << k << " (allocation " << internal_indices[k] << ")";
  }
}

// A tuple-returning callee has its leaf result allocations mapped to the
// per-leaf caller result slices keyed by ShapeIndex.
TEST_F(CompilationUnitBindingTest, MapsTupleResultLeavesToCallerSlices) {
  constexpr absl::string_view kHlo = R"(
    HloModule callee
    ENTRY e {
      p0 = f32[4] parameter(0)
      e0 = f32[4] exponential(p0)
      n0 = f32[4] negate(p0)
      ROOT t = (f32[4], f32[4]) tuple(e0, n0)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  std::unique_ptr<BufferAssignment> assignment =
      RunBufferAssignment(module.get());

  // Fabricate caller buffers a0..a3.
  BufferAllocation a0(10, 16, LogicalBuffer::Color(0));  // param 0
  BufferAllocation a1(11, 16, LogicalBuffer::Color(0));  // result leaf {0}
  BufferAllocation a2(12, 16, LogicalBuffer::Color(0));  // result leaf {1}
  BufferAllocation a3(13, 16, LogicalBuffer::Color(0));  // tuple table {}
  BufferAllocation::Slice s0(&a0, 0, 16);
  BufferAllocation::Slice s1(&a1, 0, 16);
  BufferAllocation::Slice s2(&a2, 0, 16);
  BufferAllocation::Slice s3(&a3, 0, 16);

  CalleeCallerSlices caller;
  caller.params[{0, ShapeIndex{}}] = s0;
  caller.results[ShapeIndex{0}] = s1;
  caller.results[ShapeIndex{1}] = s2;
  caller.results[ShapeIndex{}] = s3;

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<BufferAllocation::Slice> binding,
      BuildCalleeBinding(*assignment, *module->entry_computation(), caller));

  ASSERT_EQ(binding.size(), assignment->Allocations().size());

  // Every non-constant allocation is bound.
  for (const BufferAllocation& a : assignment->Allocations()) {
    if (!a.is_constant()) {
      EXPECT_NE(binding[a.index()].allocation(), nullptr)
          << "allocation " << a.index();
    }
  }

  // The two leaf result allocations map to a1 and a2.
  const HloInstruction* root =
      module->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(BufferAllocation::Slice leaf0,
                          assignment->GetUniqueSlice(root, {0}));
  TF_ASSERT_OK_AND_ASSIGN(BufferAllocation::Slice leaf1,
                          assignment->GetUniqueSlice(root, {1}));
  EXPECT_EQ(binding[leaf0.index()], s1);
  EXPECT_EQ(binding[leaf1.index()], s2);
}

// Constant allocations inside the callee are left unbound (the emitted callee
// owns its constants); params and results bind normally.
TEST_F(CompilationUnitBindingTest, LeavesConstantAllocationsUnbound) {
  constexpr absl::string_view kHlo = R"(
    HloModule callee
    ENTRY e {
      p0 = f32[64] parameter(0)
      c = f32[64] constant({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                            17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                            33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
                            49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64})
      ROOT a = f32[64] add(p0, c)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  std::unique_ptr<BufferAssignment> assignment =
      RunBufferAssignment(module.get());

  // There must be at least one constant allocation for this test to be
  // meaningful.
  bool has_constant = false;
  for (const BufferAllocation& a : assignment->Allocations()) {
    if (a.is_constant()) {
      has_constant = true;
      break;
    }
  }
  ASSERT_TRUE(has_constant) << "test needs >=1 constant allocation";

  BufferAllocation caller_p0(10, 256, LogicalBuffer::Color(0));
  BufferAllocation caller_res(12, 256, LogicalBuffer::Color(0));
  BufferAllocation::Slice p0_slice(&caller_p0, 0, 256);
  BufferAllocation::Slice res_slice(&caller_res, 0, 256);

  CalleeCallerSlices caller;
  caller.params[{0, ShapeIndex{}}] = p0_slice;
  caller.results[ShapeIndex{}] = res_slice;

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<BufferAllocation::Slice> binding,
      BuildCalleeBinding(*assignment, *module->entry_computation(), caller));

  ASSERT_EQ(binding.size(), assignment->Allocations().size());

  for (const BufferAllocation& a : assignment->Allocations()) {
    if (a.is_constant()) {
      EXPECT_EQ(binding[a.index()].allocation(), nullptr)
          << "constant allocation " << a.index() << " must be unbound";
    } else if (a.is_entry_computation_parameter()) {
      EXPECT_EQ(binding[a.index()], p0_slice);
    } else if (a.maybe_live_out()) {
      EXPECT_EQ(binding[a.index()], res_slice);
    }
  }
}

// An internal (scratch) allocation whose size exceeds the provided caller
// scratch slice is rejected.
TEST_F(CompilationUnitBindingTest, RejectsUndersizedScratchSlice) {
  constexpr absl::string_view kHlo = R"(
    HloModule callee
    ENTRY e {
      p0 = f32[64] parameter(0)
      p1 = f32[64] parameter(1)
      a = f32[64] add(p0, p1)
      b = f32[64] multiply(p0, p1)
      ROOT sub = f32[64] subtract(a, b)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  std::unique_ptr<BufferAssignment> assignment =
      RunBufferAssignment(module.get());

  // Sanity: at least one internal allocation exists to consume scratch.
  bool has_internal = false;
  for (const BufferAllocation& a : assignment->Allocations()) {
    if (!a.is_entry_computation_parameter() && !a.maybe_live_out() &&
        !a.is_constant()) {
      has_internal = true;
      break;
    }
  }
  ASSERT_TRUE(has_internal) << "test needs >=1 internal allocation";

  BufferAllocation caller_p0(10, 256, LogicalBuffer::Color(0));
  BufferAllocation caller_p1(11, 256, LogicalBuffer::Color(0));
  BufferAllocation caller_res(12, 256, LogicalBuffer::Color(0));
  BufferAllocation::Slice p0_slice(&caller_p0, 0, 256);
  BufferAllocation::Slice p1_slice(&caller_p1, 0, 256);
  BufferAllocation::Slice res_slice(&caller_res, 0, 256);

  // Provide more scratch slices than there can be internal allocations so the
  // count is never the limiter; the failure must come from undersizing (each
  // slice is only 4 bytes vs the f32[64] = 256B internal temp), not exhaustion.
  std::vector<BufferAllocation> scratch_allocs;
  scratch_allocs.reserve(4);
  for (int i = 0; i < 4; ++i) {
    scratch_allocs.emplace_back(20 + i, 4, LogicalBuffer::Color(0));
  }
  std::vector<BufferAllocation::Slice> scratch_slices;
  scratch_slices.reserve(scratch_allocs.size());
  for (BufferAllocation& alloc : scratch_allocs) {
    scratch_slices.emplace_back(&alloc, 0, 4);
  }

  CalleeCallerSlices caller;
  caller.params[{0, ShapeIndex{}}] = p0_slice;
  caller.params[{1, ShapeIndex{}}] = p1_slice;
  caller.results[ShapeIndex{}] = res_slice;
  caller.scratch = scratch_slices;

  EXPECT_THAT(
      BuildCalleeBinding(*assignment, *module->entry_computation(), caller),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("too small for")));
}

}  // namespace
}  // namespace xla::cpu
