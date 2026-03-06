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

#include "xla/service/cpu/subcomputation_fusion.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_graph.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/service/cpu/cpu_multi_output_fusion.h"

namespace xla::cpu {

absl::StatusOr<bool> SubcomputationFusionPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Build the call graph to check for kCall target sub-computations.
  auto call_graph = CallGraph::Build(module, execution_threads);

  // Check if there are any kCall target sub-computations worth pre-fusing.
  bool has_call_targets = false;
  for (const CallGraphNode& node : call_graph->nodes()) {
    HloComputation* comp = node.computation();
    if (comp->IsEntryComputation() || comp->IsFusionComputation()) {
      continue;
    }
    for (const CallSite& caller_site : node.caller_callsites()) {
      if (caller_site.instruction()->opcode() == HloOpcode::kCall) {
        has_call_targets = true;
        break;
      }
    }
    if (has_call_targets) break;
  }

  if (!has_call_targets) {
    return false;
  }

  VLOG(1) << "SubcomputationFusionPass: pre-fusing sub-computations";

  // Run CpuInstructionFusion on the full module. It processes all non-fusion
  // computations via GetNonFusionComputations(), which includes the outlined
  // sub-computations. This compacts them before inlining.
  //
  // FusionWrapper is intentionally NOT run here -- it wraps remaining unfused
  // ops into single-op fusions for the emitter. Deferring it to Phase 3
  // allows Phase 3's CpuInstructionFusion to merge raw ops across former
  // call boundaries rather than trying to merge already-wrapped fusions.
  bool changed = false;
  AliasInfo alias_info;

  {
    CpuInstructionFusion instruction_fusion(
        &alias_info, /*may_duplicate=*/!use_multi_output_fusion_);
    TF_ASSIGN_OR_RETURN(bool fusion_changed,
                        instruction_fusion.Run(module, execution_threads));
    changed |= fusion_changed;
  }

  if (use_multi_output_fusion_) {
    CpuMultiOutputFusion multi_output_fusion(&alias_info);
    TF_ASSIGN_OR_RETURN(bool mof_changed,
                        multi_output_fusion.Run(module, execution_threads));
    changed |= mof_changed;

    TupleSimplifier tuple_simplifier;
    TF_ASSIGN_OR_RETURN(bool ts_changed,
                        tuple_simplifier.Run(module, execution_threads));
    changed |= ts_changed;
  }

  VLOG(1) << "SubcomputationFusionPass: "
          << (changed ? "changed" : "no change");
  return changed;
}

}  // namespace xla::cpu
