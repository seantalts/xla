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

#include "xla/service/cpu/compilation_unit_scratch_rewriter.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/hlo_module_stitcher.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/compilation_unit_binding.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

std::vector<const BufferAllocation*> GetCalleeScratchAllocations(
    const BufferAssignment& callee_assignment) {
  std::vector<const BufferAllocation*> scratch;
  for (const BufferAllocation& allocation : callee_assignment.Allocations()) {
    if (IsCalleeInternalAllocation(allocation)) {
      scratch.push_back(&allocation);
    }
  }
  return scratch;
}

absl::StatusOr<bool> RewriteCompilationUnitScratch(
    HloModule* parent,
    const absl::flat_hash_map<std::string, const BufferAssignment*>&
        submodule_assignments) {
  bool changed = false;

  for (HloComputation* comp : parent->MakeComputationPostOrder()) {
    // Collect the marked multi-module custom-calls first so we don't re-find
    // the freshly-created ones (which carry the same custom-call target).
    std::vector<HloCustomCallInstruction*> sites;
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kCustomCall) {
        continue;
      }
      auto* cc = Cast<HloCustomCallInstruction>(inst);
      if (cc->custom_call_target() != kMultiModuleCustomCallTarget) {
        continue;
      }
      sites.push_back(cc);
    }

    for (HloCustomCallInstruction* cc : sites) {
      const std::string& submodule = cc->raw_backend_config_string();
      auto it = submodule_assignments.find(submodule);
      if (it == submodule_assignments.end()) {
        return absl::NotFoundError(absl::StrCat(
            "No buffer assignment for compilation-unit submodule '", submodule,
            "'"));
      }

      std::vector<const BufferAllocation*> scratch =
          GetCalleeScratchAllocations(*it->second);
      if (scratch.empty()) {
        continue;
      }

      // We move the original result into tuple index {0} of the new
      // custom-call. Any output_to_operand_aliasing on the original instruction
      // is expressed relative to the old (non-tuple) output shape and cannot be
      // correctly remapped here, so refuse rather than silently produce wrong
      // aliasing.
      if (!cc->output_to_operand_aliasing().empty()) {
        return absl::UnimplementedError(absl::StrCat(
            "Cannot rewrite compilation-unit custom-call '", cc->name(),
            "' with output_to_operand_aliasing: moving the result into a tuple "
            "would invalidate the aliasing."));
      }

      std::vector<Shape> tuple_shapes;
      tuple_shapes.reserve(1 + scratch.size());
      tuple_shapes.push_back(cc->shape());
      for (const BufferAllocation* allocation : scratch) {
        tuple_shapes.push_back(
            ShapeUtil::MakeShape(S8, {allocation->size()}));
      }
      Shape new_shape = ShapeUtil::MakeTupleShape(tuple_shapes);

      // Clone so every attribute (metadata, backend config, api_version,
      // side-effect flag, frontend attributes, sharding, ...) is carried over;
      // only the shape changes. Building the custom-call by hand would silently
      // drop anything not explicitly copied.
      HloInstruction* new_cc = comp->AddInstruction(
          cc->CloneWithNewOperands(new_shape, cc->operands()));

      HloInstruction* gte = comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(cc->shape(), new_cc, 0));

      TF_RETURN_IF_ERROR(cc->ReplaceAllUsesWith(gte));
      if (comp->root_instruction() == cc) {
        comp->set_root_instruction(gte);
      }
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(cc));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla::cpu
