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

#include "xla/backends/cpu/codegen/emitters/region_kernel_emitter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

RegionKernelEmitter::RegionKernelEmitter(
    const HloInstruction* call, const BufferAssignment* buffer_assignment,
    mlir::MLIRContext* mlir_context)
    : call_(call),
      region_(call->to_apply()),
      buffer_assignment_(buffer_assignment),
      mlir_context_(mlir_context) {}

// An Increment-1' region is supported iff:
//   - the root produces a single array (non-tuple) value, and
//   - no member is control flow / scatter / tuple / unsupported op.
//
// Unlike the elemental scaffold this does NOT require every member to share the
// root shape: the region is reified as a fusion view and emitted via the xtile
// tiled emitter, whose `EmitTiledComputation` threads differently-shaped
// intermediate tensors through the schedule. So dot/reduce/broadcast chains are
// admitted. Scatter and control flow are Increment 2/3 and decline cleanly.
bool RegionKernelEmitter::IsSupportedRegion(const HloInstruction* call) {
  const HloComputation* region = call->to_apply();
  const HloInstruction* root = region->root_instruction();
  // Multi-output (tuple-root) regions are deferred: the tiled emitter requires
  // a single array root.
  if (!root->shape().IsArray()) {
    return false;
  }
  for (const HloInstruction* instr : region->instructions()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      continue;
    }
    switch (instr->opcode()) {
      // Control flow / scatter / nested calls / non-tileable runtime ops:
      // declined; later increments.
      case HloOpcode::kWhile:
      case HloOpcode::kConditional:
      case HloOpcode::kScatter:
      case HloOpcode::kCall:
      case HloOpcode::kCustomCall:
      case HloOpcode::kSort:
      case HloOpcode::kFft:
      case HloOpcode::kInfeed:
      case HloOpcode::kOutfeed:
      // Tuple plumbing inside a region implies a multi-output structure the
      // single-array-root tiled emitter cannot model yet.
      case HloOpcode::kTuple:
      case HloOpcode::kGetTupleElement:
        return false;
      default:
        break;
    }
  }
  return true;
}

absl::StatusOr<const HloFusionInstruction*>
RegionKernelEmitter::BuildFusionView(
    std::unique_ptr<HloModule>& view_module) const {
  // The fusion view lives in a throwaway module that copies the original
  // debug options and additionally turns on tiling propagation, so the tiled
  // emitter routes the region through the `TilingSpace` path that assigns ALL
  // tiling-space dims (the dot contracting dim included) -- which the
  // root-only `GetTiling` default path does not.
  HloModuleConfig config = call_->GetModule()->config();
  DebugOptions debug_options = config.debug_options();
  debug_options.set_xla_cpu_experimental_enable_tiling_propagation(true);
  config.set_debug_options(debug_options);
  view_module = std::make_unique<HloModule>(
      absl::StrCat(call_->name(), "_region_view"), config);

  const HloComputation* region = region_;

  // STEP 1: flatten nested fusions. Live (post-instruction-fusion) regions
  // contain nested `kLoop` fusions that `SymbolicTileAnalysis` / the tiling
  // propagation path reject. We first clone the region into a standalone
  // computation in the view module, then `Defuse()` every `kFusion` member to
  // a fixpoint so the body is reduced to raw ops. `Defuse()` clones the fused
  // body into the parent computation, rewires uses, and removes the fusion;
  // applying it to a fixpoint un-nests arbitrarily deep fusion trees. Reduce
  // reducers and other `to_apply` subcomputations are preserved as called
  // computations (they are not fusions, so Defuse leaves them be).
  HloComputation* flat_region = nullptr;
  {
    HloCloneContext flatten_context(view_module.get());
    HloComputation::Builder flat_builder(
        absl::StrCat(call_->name(), "_region_flat"));
    absl::flat_hash_map<const HloInstruction*, HloInstruction*> flat_map;
    for (int64_t i = 0; i < region->num_parameters(); ++i) {
      const HloInstruction* region_param = region->parameter_instruction(i);
      flat_map[region_param] =
          flat_builder.AddInstruction(HloInstruction::CreateParameter(
              i, region_param->shape(), absl::StrCat("p", i)));
    }
    HloInstruction* flat_root = nullptr;
    for (const HloInstruction* instr : region->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kParameter) {
        continue;
      }
      std::vector<HloInstruction*> new_operands;
      new_operands.reserve(instr->operand_count());
      for (const HloInstruction* operand : instr->operands()) {
        new_operands.push_back(flat_map.at(operand));
      }
      HloInstruction* cloned = flat_builder.AddInstruction(
          instr->CloneWithNewOperands(instr->shape(), new_operands,
                                      &flatten_context));
      flat_map[instr] = cloned;
      if (instr == region->root_instruction()) {
        flat_root = cloned;
      }
    }
    if (flat_root == nullptr) {
      return absl::InternalError("Region root not found while cloning.");
    }
    flat_region =
        view_module->AddEmbeddedComputation(flat_builder.Build(flat_root));

    // Defuse to a fixpoint: each Defuse can expose freshly-cloned inner
    // fusions, so re-scan until no kFusion remains.
    bool changed = true;
    while (changed) {
      changed = false;
      for (HloInstruction* instr : flat_region->MakeInstructionPostOrder()) {
        if (instr->opcode() == HloOpcode::kFusion) {
          RETURN_IF_ERROR(instr->Defuse());
          changed = true;
          break;
        }
      }
    }
  }

  // Build the FUSED computation (the body): one parameter per kCall operand in
  // kCall operand order, then a clone of every now-flat region member with
  // operands remapped to those parameters. The clone uses a context targeting
  // `view_module`, so the region's nested called computations (e.g. reduce
  // reducers) are cloned in too. Constructing the fused computation directly
  // (rather than fusing live instructions) lets us pin the fusion operand
  // order to the kCall operand order, so the kernel argument buffers --
  // resolved against the kCall by GetKernelSpec -- line up exactly.
  HloCloneContext clone_context(view_module.get());
  HloComputation::Builder fused_builder(
      absl::StrCat(call_->name(), "_region_fused"));
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> map;
  for (int64_t i = 0; i < flat_region->num_parameters(); ++i) {
    const HloInstruction* region_param =
        flat_region->parameter_instruction(i);
    map[region_param] =
        fused_builder.AddInstruction(HloInstruction::CreateParameter(
            i, region_param->shape(), absl::StrCat("p", i)));
  }
  HloInstruction* fused_root = nullptr;
  for (const HloInstruction* instr : flat_region->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(instr->operand_count());
    for (const HloInstruction* operand : instr->operands()) {
      new_operands.push_back(map.at(operand));
    }
    HloInstruction* cloned = fused_builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), new_operands,
                                    &clone_context));
    map[instr] = cloned;
    if (instr == flat_region->root_instruction()) {
      fused_root = cloned;
    }
  }
  if (fused_root == nullptr) {
    return absl::InternalError("Region root not found while cloning.");
  }
  HloComputation* fused_computation = view_module->AddEmbeddedComputation(
      fused_builder.Build(fused_root));

  // Build the entry computation: one parameter per kCall operand (in order),
  // plus the fusion instruction whose operands are exactly those parameters in
  // order. The fused_computation's parameters correspond positionally.
  HloComputation::Builder entry_builder(
      absl::StrCat(call_->name(), "_region_entry"));
  std::vector<HloInstruction*> fusion_operands;
  fusion_operands.reserve(flat_region->num_parameters());
  for (int64_t i = 0; i < flat_region->num_parameters(); ++i) {
    fusion_operands.push_back(
        entry_builder.AddInstruction(HloInstruction::CreateParameter(
            i, flat_region->parameter_instruction(i)->shape(),
            absl::StrCat("p", i))));
  }
  HloInstruction* fusion =
      entry_builder.AddInstruction(HloInstruction::CreateFusion(
          fused_computation->root_instruction()->shape(),
          HloInstruction::FusionKind::kLoop, fusion_operands,
          fused_computation));
  view_module->AddEntryComputation(entry_builder.Build(fusion));

  return Cast<HloFusionInstruction>(fusion);
}

absl::StatusOr<RegionKernelEmitter::KernelDefinition>
RegionKernelEmitter::EmitKernelDefinition() {
  if (!IsSupportedRegion(call_)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "RegionKernelEmitter declines region for ", call_->name(),
        ": only straight-line single-array-output regions (no scatter, no "
        "control flow, no tuples) are supported; falling back to the legacy "
        "ComputationKernelEmitter."));
  }

  std::unique_ptr<HloModule> view_module;
  ASSIGN_OR_RETURN(const HloFusionInstruction* fusion_view,
                   BuildFusionView(view_module));

  // Single-workgroup by design: a region is only formed when the work is too
  // small to parallelize.
  TiledEmissionResult result = EmitTiledRegionKernel(
      *mlir_context_, *fusion_view, *call_, buffer_assignment_, call_->name(),
      /*num_work_groups=*/1);

  if (!result.kernel.ok()) {
    return absl::UnimplementedError(absl::StrCat(
        "RegionKernelEmitter: tiled emission declined region ", call_->name(),
        ": ", result.kernel.status().message(),
        ". Falling back to legacy path."));
  }
  return std::move(*result.kernel);
}

}  // namespace xla::cpu
