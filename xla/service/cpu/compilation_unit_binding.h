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

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Builds the per-allocation binding consumed by a RemappedCallThunk for one call
// site of a shared `compilation_unit` callee.
//
// The callee is emitted once against its OWN (private) buffer assignment, whose
// allocations are indexed 0..N-1. This returns a vector indexed by that private
// allocation index, giving the caller buffer that provides each callee
// allocation at this call site:
//   - a callee parameter allocation p  -> `caller_param_slices[p]`
//   - the callee result allocation     -> `caller_result_slice`
//   - an internal (scratch) allocation -> not yet supported (see plan 3B)
absl::StatusOr<std::vector<BufferAllocation::Slice>> BuildCalleeBinding(
    const BufferAssignment& callee_assignment,
    absl::Span<const BufferAllocation::Slice> caller_param_slices,
    const BufferAllocation::Slice& caller_result_slice);

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_COMPILATION_UNIT_BINDING_H_
