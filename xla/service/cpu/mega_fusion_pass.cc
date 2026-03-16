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

#include "xla/service/cpu/mega_fusion_pass.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace cpu {

// An instruction is fusible if it's elementwise or one of the data-movement
// ops that the MLIR elemental emitter supports. This mirrors the set used by
// CanBeLoopFused in cpu_instruction_fusion.cc plus the elementwise ops wrapped
// by FusionWrapper.
bool MegaFusionPass::CanFuse(const HloInstruction& instruction) {
  // Don't re-fuse already-fused instructions.
  if (instruction.opcode() == HloOpcode::kFusion) {
    return false;
  }
  // Parameters and constants are absorbed as fusion operands, not fused
  // as computation instructions.
  if (instruction.opcode() == HloOpcode::kParameter ||
      instruction.opcode() == HloOpcode::kConstant) {
    return false;
  }
  // Tuple / get-tuple-element are structural, not fusible.
  if (instruction.opcode() == HloOpcode::kTuple ||
      instruction.opcode() == HloOpcode::kGetTupleElement) {
    return false;
  }

  if (instruction.IsElementwise()) {
    return true;
  }

  switch (instruction.opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kIota:
    case HloOpcode::kPad:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<bool> MegaFusionPass::FuseComputation(
    HloComputation* computation) {
  bool changed = false;

  // Iterate in reverse post-order (consumers before producers) so we create
  // fusions starting from the "output" side and absorb producers into them.
  auto post_order = computation->MakeInstructionPostOrder();

  // Track which instructions have already been absorbed into a fusion.
  absl::flat_hash_set<const HloInstruction*> fused;

  // Process in reverse post-order: from outputs toward inputs.
  for (int i = static_cast<int>(post_order.size()) - 1; i >= 0; --i) {
    HloInstruction* instruction = post_order[i];

    // Skip if already fused, or not fusible, or already a fusion.
    if (fused.contains(instruction) || !CanFuse(*instruction)) {
      continue;
    }

    // Create a seed fusion from this instruction.
    auto* fusion = computation->AddInstruction(HloInstruction::CreateFusion(
        instruction->shape(), HloInstruction::FusionKind::kLoop, instruction));

    // Transfer metadata.
    if (computation->parent() && computation->parent()->has_schedule()) {
      computation->parent()->schedule().replace_instruction(computation,
                                                            instruction,
                                                            fusion);
    }
    TF_RETURN_IF_ERROR(fusion->CopyAllControlDepsFrom(instruction));
    TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());
    TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(fusion));
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
    fused.insert(instruction);
    changed = true;

    // BFS: greedily fuse fusible operands into this fusion.
    // An operand is fusible if:
    //  1. It passes CanFuse()
    //  2. All of its users are already inside this fusion (single-consumer
    //     constraint to avoid duplicating computation).
    bool made_progress = true;
    while (made_progress) {
      made_progress = false;
      // Iterate over current fusion operands (parameters from outside).
      for (int op_idx = 0; op_idx < fusion->operand_count(); ++op_idx) {
        HloInstruction* operand = fusion->mutable_operand(op_idx);
        if (fused.contains(operand) || !CanFuse(*operand)) {
          continue;
        }
        // Only fuse if all users of this operand are inside the fusion
        // (i.e., the operand has exactly one user: this fusion instruction).
        if (operand->user_count() != 1) {
          continue;
        }

        // Fuse the operand into the fusion.
        fusion->FuseInstruction(operand);
        fused.insert(operand);
        made_progress = true;

        // If the operand is now dead (no users outside fusion), remove it.
        if (operand->user_count() == 0) {
          TF_RETURN_IF_ERROR(operand->DropAllControlDeps());
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(operand));
        }
        // Restart the operand scan since fusion operands changed.
        break;
      }
    }

    // Name the fusion for debugging.
    if (computation->parent()) {
      computation->parent()->SetAndUniquifyInstrName(fusion, "mega_fusion");
      computation->parent()->SetAndUniquifyComputationName(
          fusion->fused_instructions_computation(),
          "mega_fusion_computation");
    }
  }

  return changed;
}

absl::StatusOr<bool> MegaFusionPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    // Skip fusion computations (the body of existing fusions).
    if (computation->IsFusionComputation()) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool computation_changed, FuseComputation(computation));
    changed |= computation_changed;
  }
  return changed;
}

}  // namespace cpu
}  // namespace xla
