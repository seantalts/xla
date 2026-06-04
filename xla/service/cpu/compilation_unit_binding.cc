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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

absl::StatusOr<std::vector<BufferAllocation::Slice>> BuildCalleeBinding(
    const BufferAssignment& callee_assignment,
    absl::Span<const BufferAllocation::Slice> caller_param_slices,
    const BufferAllocation::Slice& caller_result_slice,
    absl::Span<const BufferAllocation::Slice> caller_scratch_slices) {
  const std::vector<BufferAllocation>& allocations =
      callee_assignment.Allocations();
  std::vector<BufferAllocation::Slice> binding(allocations.size());

  // Internal allocations consume scratch slices in iteration order, which is
  // ascending allocation index (Allocations() is index-ordered). The producer
  // that reserves the scratch tuple elements must use the same order.
  size_t scratch_index = 0;
  for (const BufferAllocation& allocation : allocations) {
    // Check parameters before live-out so a parameter aliased with the output
    // binds to the caller's operand buffer. (TODO: revisit in/out aliasing and
    // constant allocations inside a unit.)
    if (allocation.is_entry_computation_parameter()) {
      int64_t parameter = allocation.parameter_number();
      if (parameter < 0 ||
          parameter >= static_cast<int64_t>(caller_param_slices.size())) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Callee parameter ", parameter, " has no caller slice (",
            caller_param_slices.size(), " provided)"));
      }
      binding[allocation.index()] = caller_param_slices[parameter];
    } else if (allocation.maybe_live_out()) {
      binding[allocation.index()] = caller_result_slice;
    } else {
      if (scratch_index >= caller_scratch_slices.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Callee has more internal allocations than caller scratch slices (",
            caller_scratch_slices.size(), " provided)"));
      }
      binding[allocation.index()] = caller_scratch_slices[scratch_index++];
    }
  }
  return binding;
}

}  // namespace xla::cpu
