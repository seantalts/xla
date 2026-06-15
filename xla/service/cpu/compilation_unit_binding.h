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

#ifndef XLA_SERVICE_CPU_COMPILATION_UNIT_BINDING_H_
#define XLA_SERVICE_CPU_COMPILATION_UNIT_BINDING_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"

namespace xla::cpu {

// True iff `allocation` is a callee-internal (scratch) allocation: not an entry
// parameter, not live-out, and not a constant. These are the allocations that
// BuildCalleeBinding binds to per-site scratch slices and that the 3B scratch
// rewriter reserves space for. Centralized here so the binding and the scratch
// rewriter agree on exactly one definition of "internal".
inline bool IsCalleeInternalAllocation(const BufferAllocation& allocation) {
  return !allocation.is_entry_computation_parameter() &&
         !allocation.maybe_live_out() && !allocation.is_constant();
}

// The caller-side buffer slices that provide a shared `compilation_unit`
// callee's inputs, outputs, and scratch at one call site. Used to map the
// callee's private allocations onto the caller's buffers.
struct CalleeCallerSlices {
  // For each callee entry parameter, keyed by (parameter_number, shape index
  // within that parameter), the caller slice that provides it. Tuple-shaped
  // parameters contribute one entry per leaf (and possibly the tuple table).
  absl::flat_hash_map<std::pair<int64_t, ShapeIndex>, BufferAllocation::Slice>
      params;
  // For each shape index of the callee root, the caller slice that receives
  // that output. Tuple-shaped results contribute one entry per leaf, plus the
  // top-level tuple table at index {} if the callee materializes one.
  absl::flat_hash_map<ShapeIndex, BufferAllocation::Slice> results;
  // Per-site scratch slices, consumed by the callee's internal allocations in
  // order of increasing callee allocation index (see plan 3B: tuple output
  // `(result, scratch...)`).
  std::vector<BufferAllocation::Slice> scratch;
};

// Builds the per-allocation binding consumed by a RemappedCallThunk for one call
// site of a shared `compilation_unit` callee.
//
// The callee is emitted once against its OWN (private) buffer assignment, whose
// allocations are indexed 0..N-1. This returns a vector indexed by that private
// allocation index, giving the caller buffer that provides each callee
// allocation at this call site:
//   - a callee parameter allocation     -> `caller.params[{param, index}]`
//   - a callee result allocation (leaf
//     or tuple table)                   -> the matching `caller.results[index]`
//   - an internal (scratch) allocation  -> the next `caller.scratch` entry,
//     consumed in order of increasing callee allocation index.
//   - a constant allocation             -> left unbound (null allocation); the
//     emitted callee owns its own constants.
absl::StatusOr<std::vector<BufferAllocation::Slice>> BuildCalleeBinding(
    const BufferAssignment& callee_assignment,
    const HloComputation& callee_entry, const CalleeCallerSlices& caller);

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_COMPILATION_UNIT_BINDING_H_
