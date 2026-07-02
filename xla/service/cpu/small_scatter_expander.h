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

#ifndef XLA_SERVICE_CPU_SMALL_SCATTER_EXPANDER_H_
#define XLA_SERVICE_CPU_SMALL_SCATTER_EXPANDER_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/scatter_expander.h"

namespace xla::cpu {

// Expands *small* scatters into while loops of gather/dynamic-update-slice,
// leaving large scatters to the dedicated fusion-emitter scatter kernel.
//
// Why: small scatters fragment hoistable regions. SmallRegionHoistingPass
// treats scatter as a region boundary because the legacy kernel emitter
// cannot emit it, so a scatter-heavy small model (e.g. jaxley, jax #26145)
// degenerates into many tiny regions whose per-thunk dispatch cost dominates
// the useful compute. Expanding a small scatter into a while+gather+DUS loop
// turns it into ordinary region-eligible control flow, letting region
// hoisting fold the surrounding program into a single kernel. For large
// scatters the dedicated MLIR scatter fusion kernel wins, so they are left
// untouched.
//
// A scatter is "small" when the sum of the byte sizes of all its operand
// shapes and all its result shapes (per tuple leaf, for variadic scatters)
// is below `small_buffer_access_size`.
class SmallScatterExpander final : public ScatterExpander {
 public:
  explicit SmallScatterExpander(int64_t small_buffer_access_size)
      : ScatterExpander(kEliminateAllScatters),
        small_buffer_access_size_(small_buffer_access_size) {}

  absl::string_view name() const override { return "small-scatter-expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* inst) override;

 private:
  int64_t small_buffer_access_size_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_SMALL_SCATTER_EXPANDER_H_
