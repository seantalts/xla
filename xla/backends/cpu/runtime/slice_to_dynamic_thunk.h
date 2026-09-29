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

#ifndef XLA_BACKENDS_CPU_RUNTIME_SLICE_TO_DYNAMIC_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_SLICE_TO_DYNAMIC_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Implements the "SliceToDynamic" custom call: packs the dynamic box of a
// statically shaped source array densely into the destination buffer and
// appends the dynamic extents as int32 metadata after the static payload.
//
// Destination buffer layout:
//   [0, raw_data_size)                 packed elements
//   [raw_data_size, raw_data_size + 4 * rank)  int32 extent per dimension
class SliceToDynamicThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<SliceToDynamicThunk>> Create(
      Info info, ShapedSlice source,
      std::vector<BufferAllocation::Slice> dim_sizes, ShapedSlice destination);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

  const ShapedSlice& source() const { return source_; }
  absl::Span<const BufferAllocation::Slice> dim_sizes() const {
    return dim_sizes_;
  }
  const ShapedSlice& destination() const { return destination_; }

 private:
  SliceToDynamicThunk(Info info, ShapedSlice source,
                      std::vector<BufferAllocation::Slice> dim_sizes,
                      ShapedSlice destination);

  ShapedSlice source_;
  std::vector<BufferAllocation::Slice> dim_sizes_;
  ShapedSlice destination_;

  int64_t element_size_;
  int64_t raw_data_size_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_SLICE_TO_DYNAMIC_THUNK_H_
