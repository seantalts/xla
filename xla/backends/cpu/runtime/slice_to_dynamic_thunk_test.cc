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

#include "xla/backends/cpu/runtime/slice_to_dynamic_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

using ::testing::HasSubstr;

struct Case {
  Case(std::string name, Literal source, std::vector<bool> dynamic_dims,
       std::vector<int32_t> extents, std::vector<int32_t> expected)
      : name(std::move(name)),
        source(std::make_shared<Literal>(std::move(source))),
        dynamic_dims(std::move(dynamic_dims)),
        extents(std::move(extents)),
        expected(std::move(expected)) {}

  std::string name;
  std::shared_ptr<const Literal> source;
  std::vector<bool> dynamic_dims;
  std::vector<int32_t> extents;
  std::vector<int32_t> expected;
};

class SliceToDynamicThunkTest : public ::testing::TestWithParam<Case> {
 protected:
  absl::StatusOr<std::unique_ptr<SliceToDynamicThunk>> CreateThunk(
      const Literal& source, const std::vector<bool>& dynamic_dims,
      absl::Span<const int32_t> extents, int64_t destination_words) {
    Shape dynamic_shape = source.shape();
    for (size_t i = 0; i < dynamic_dims.size(); ++i) {
      dynamic_shape.set_dynamic_dimension(i, dynamic_dims[i]);
    }
    literals_.push_back(source.Clone());
    for (int32_t extent : extents) {
      literals_.push_back(LiteralUtil::CreateR0<int32_t>(extent));
    }
    literals_.push_back(
        LiteralUtil::CreateFull<int32_t>({destination_words}, 0));
    for (size_t i = 0; i < literals_.size(); ++i) {
      allocs_.push_back(CreateBufferAllocation(i, literals_[i]));
    }
    for (const BufferAllocation& alloc : allocs_) {
      slices_.push_back(CreateBufferAllocationSlice(alloc));
    }
    return SliceToDynamicThunk::Create(
        {"slice_to_dynamic"}, ShapedSlice{slices_[0], source.shape()},
        {slices_.begin() + 1, slices_.end() - 1},
        ShapedSlice{slices_.back(), dynamic_shape});
  }

  absl::StatusOr<std::vector<int32_t>> Run(
      const Literal& source, const std::vector<bool>& dynamic_dims,
      absl::Span<const int32_t> extents) {
    int64_t words = ShapeUtil::ByteSizeOf(source.shape()) / sizeof(int32_t);
    ABSL_ASSIGN_OR_RETURN(auto thunk, CreateThunk(source, dynamic_dims, extents,
                                                  words + extents.size()));
    std::vector<Literal*> literals;
    for (Literal& literal : literals_) {
      literals.push_back(&literal);
    }
    BufferAllocations allocations =
        CreateBufferAllocations(absl::MakeSpan(literals));
    Thunk::ExecuteParams params = {nullptr, &allocations};
    auto execute_event = thunk->Execute(params);
    tsl::BlockUntilReady(execute_event);
    if (execute_event.IsError()) {
      return execute_event.GetError();
    }
    absl::Span<const int32_t> data = literals_.back().data<int32_t>();
    return std::vector<int32_t>(data.begin(), data.end());
  }

  std::vector<Literal> literals_;
  std::vector<BufferAllocation> allocs_;
  std::vector<BufferAllocation::Slice> slices_;
};

Literal Source2x3() {
  return LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}});
}

TEST_P(SliceToDynamicThunkTest, Packs) {
  const Case& c = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto actual,
                          Run(*c.source, c.dynamic_dims, c.extents));
  EXPECT_EQ(actual, c.expected);
}

INSTANTIATE_TEST_SUITE_P(
    SliceToDynamicThunk, SliceToDynamicThunkTest,
    ::testing::Values(
        Case("fully_dynamic_2d", Source2x3(), {true, true}, {2, 2},
             {1, 2, 4, 5, 0, 0, 2, 2}),
        Case("partially_dynamic_2d",
             LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}, {5, 6}}),
             {true, false}, {2, 2}, {1, 2, 3, 4, 0, 0, 2, 2}),
        Case("dim0_minor_layout",
             LiteralUtil::CreateR2WithLayout<int32_t>(
                 {{1, 2, 3}, {4, 5, 6}}, LayoutUtil::MakeLayout({0, 1})),
             {true, true}, {2, 2}, {1, 4, 2, 5, 0, 0, 2, 2}),
        Case("zero_extent", Source2x3(), {true, false}, {0, 3},
             {0, 0, 0, 0, 0, 0, 0, 3}),
        Case("rank1", LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4}), {true}, {3},
             {1, 2, 3, 0, 3})),
    [](const auto& info) { return info.param.name; });

TEST_F(SliceToDynamicThunkTest, ExtentBeyondBoundIsAnError) {
  EXPECT_THAT(Run(Source2x3(), {true, true}, {3, 3}).status().message(),
              HasSubstr("outside [0, 2]"));
}

TEST_F(SliceToDynamicThunkTest, RejectsShortDestination) {
  EXPECT_THAT(
      CreateThunk(Source2x3(), {true, true}, {2, 3}, /*destination_words=*/6)
          .status()
          .message(),
      HasSubstr("needs 32"));
}

TEST_F(SliceToDynamicThunkTest, BufferUses) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CreateThunk(Source2x3(), {true, true}, {2, 3}, /*destination_words=*/8));
  Thunk::BufferUses uses = thunk->buffer_uses();
  ASSERT_EQ(uses.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(uses[i].slice(), slices_[i]);
    EXPECT_EQ(uses[i].access(), i == 3 ? BufferUse::MemoryAccess::kWrite
                                       : BufferUse::MemoryAccess::kRead);
  }
}

}  // namespace
}  // namespace xla::cpu
