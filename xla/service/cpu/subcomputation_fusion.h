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

#ifndef XLA_SERVICE_CPU_SUBCOMPUTATION_FUSION_H_
#define XLA_SERVICE_CPU_SUBCOMPUTATION_FUSION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::cpu {

// Runs CpuInstructionFusion (and optionally CpuMultiOutputFusion) on each
// non-entry sub-computation independently, in bottom-up callgraph order
// (callees before callers). This pre-compacts sub-computations so that
// subsequent full inlining + fusion operates on a smaller graph.
//
// This is Phase 1 of the two-phase fusion design: fuse within each
// sub-computation first, then inline, then fuse again across boundaries.
// FusionWrapper is intentionally deferred to Phase 3 so that Phase 3 can
// merge raw fusions rather than trying to merge already-wrapped ops.
class SubcomputationFusionPass : public HloModulePass {
 public:
  explicit SubcomputationFusionPass(bool use_multi_output_fusion)
      : use_multi_output_fusion_(use_multi_output_fusion) {}

  absl::string_view name() const override {
    return "subcomputation-fusion";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;

 private:
  bool use_multi_output_fusion_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_SUBCOMPUTATION_FUSION_H_
