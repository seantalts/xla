/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_MEGA_FUSION_PASS_H_
#define XLA_SERVICE_CPU_MEGA_FUSION_PASS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace cpu {

// MegaFusionPass greedily fuses all elementally-emittable instructions in each
// computation into maximal kLoop fusions. This is designed for the O1 (fast
// compile) pipeline where we want to minimize the number of kernel launches
// while ensuring all fused ops go through the MLIR LoopFusionKernelEmitter.
//
// Unlike CpuInstructionFusion (which does pairwise producer-consumer fusion
// with profitability heuristics) or FusionWrapper (which wraps single ops),
// this pass creates the largest possible fusions by merging connected
// subgraphs of fusible ops.
class MegaFusionPass : public HloModulePass {
 public:
  absl::string_view name() const override { return "mega-fusion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;

 private:
  // Returns true if the instruction can be fused into a mega-fusion.
  static bool CanFuse(const HloInstruction& instruction);

  // Process a single computation, creating mega-fusions.
  absl::StatusOr<bool> FuseComputation(HloComputation* computation);
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_MEGA_FUSION_PASS_H_
