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

#ifndef XLA_SERVICE_CPU_COMPILATION_UNIT_SCRATCH_REWRITER_H_
#define XLA_SERVICE_CPU_COMPILATION_UNIT_SCRATCH_REWRITER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Callee-internal allocations (non-parameter, non-live-out, non-constant) in
// increasing allocation-index order — the order BuildCalleeBinding consumes
// scratch in.
std::vector<const BufferAllocation*> GetCalleeScratchAllocations(
    const BufferAssignment& callee_assignment);

// Rewrites every _xla_multi_module_call custom-call in `parent` whose callee
// (by backend_config submodule name) has internal allocations: result shape S
// becomes (S, s8[n_0], ..., s8[n_k]); a GTE(0) replaces all uses. Returns true
// if any site changed. Must run BEFORE `parent` is scheduled.
absl::StatusOr<bool> RewriteCompilationUnitScratch(
    HloModule* parent,
    const absl::flat_hash_map<std::string, const BufferAssignment*>&
        submodule_assignments);

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_COMPILATION_UNIT_SCRATCH_REWRITER_H_
