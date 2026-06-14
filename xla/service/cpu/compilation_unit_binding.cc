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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::cpu {

absl::StatusOr<std::vector<BufferAllocation::Slice>> BuildCalleeBinding(
    const BufferAssignment& callee_assignment,
    const HloComputation& callee_entry, const CalleeCallerSlices& caller) {
  // Build a map from callee result allocation index to the caller result slice
  // that receives it, by walking every subshape of the callee root. Several
  // shape indices may share one allocation (e.g. a tuple table); the map must
  // be consistent across them.
  const HloInstruction* root = callee_entry.root_instruction();
  absl::flat_hash_map<int64_t, BufferAllocation::Slice> result_binding;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      root->shape(),
      [&](const Shape& /*subshape*/, const ShapeIndex& index) -> absl::Status {
        if (!callee_assignment.HasAllocationAt(root, index)) {
          // Genuinely no buffer at this shape index (e.g. an interior tuple node
          // without a materialized table); nothing to bind. Distinct from an
          // ambiguous slice, which GetUniqueSlice also reports as an error but
          // must NOT be swallowed below.
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            callee_assignment.GetUniqueSlice(root, index));
        auto it = caller.results.find(index);
        if (it == caller.results.end()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Callee result at shape index ", index.ToString(),
                           " has no caller result slice"));
        }
        auto [inserted_it, inserted] =
            result_binding.try_emplace(slice.index(), it->second);
        if (!inserted && inserted_it->second != it->second) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Callee result allocation ", slice.index(),
              " maps to conflicting caller result slices"));
        }
        return absl::OkStatus();
      }));

  const std::vector<BufferAllocation>& allocations =
      callee_assignment.Allocations();
  std::vector<BufferAllocation::Slice> binding(allocations.size());

  // Internal allocations consume scratch slices in iteration order, which is
  // ascending allocation index (Allocations() is index-ordered). The producer
  // that reserves the scratch tuple elements must use the same order.
  size_t next_scratch = 0;
  for (const BufferAllocation& allocation : allocations) {
    // Constant allocations are owned by the emitted callee; leave them unbound.
    if (allocation.is_constant()) {
      continue;
    }
    // Check parameters before live-out so a parameter aliased with the output
    // binds to the caller's operand buffer. (TODO: revisit in/out aliasing and
    // constant allocations inside a unit.)
    if (allocation.is_entry_computation_parameter()) {
      std::pair<int64_t, ShapeIndex> key{allocation.parameter_number(),
                                         allocation.param_shape_index()};
      auto it = caller.params.find(key);
      if (it == caller.params.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Callee parameter ", allocation.parameter_number(),
            " at shape index ", allocation.param_shape_index().ToString(),
            " has no caller slice"));
      }
      binding[allocation.index()] = it->second;
      continue;
    }
    if (auto it = result_binding.find(allocation.index());
        it != result_binding.end()) {
      binding[allocation.index()] = it->second;
      continue;
    }
    if (allocation.maybe_live_out()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Callee live-out allocation ", allocation.index(),
          " is not reachable from any caller result slice"));
    }
    // Internal (scratch) allocation.
    if (next_scratch >= caller.scratch.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Callee has more internal allocations than caller scratch slices (",
          caller.scratch.size(), " provided)"));
    }
    const BufferAllocation::Slice& scratch = caller.scratch[next_scratch++];
    if (scratch.size() < allocation.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Caller scratch slice of ", scratch.size(),
          " bytes is too small for callee internal allocation ",
          allocation.index(), " of ", allocation.size(), " bytes"));
    }
    binding[allocation.index()] = scratch;
  }
  return binding;
}

}  // namespace xla::cpu
