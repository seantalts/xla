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

#include "xla/service/cpu/small_region_hoisting_pass.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

// An instruction inherently needs runtime services and cannot live inside a
// single-kernel region. These are the region boundaries.
bool InstructionIsUnavailable(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kCustomCall:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kScatter:
    case HloOpcode::kSort:
    case HloOpcode::kFft:
    case HloOpcode::kPartitionId:
    case HloOpcode::kReplicaId:
      return true;
    // Legacy call emitter does not support custom fusions (library fusions).
    case HloOpcode::kFusion:
      return instr->fusion_kind() == HloInstruction::FusionKind::kCustom;
    default:
      return IsCollective(instr);
  }
}

// True if `instr` or any instruction reachable through its called computations
// is unavailable. Memoized across the pass.
bool ContainsUnavailableInstruction(
    const HloInstruction* instr,
    absl::flat_hash_map<const HloInstruction*, bool>& memo) {
  if (auto it = memo.find(instr); it != memo.end()) {
    return it->second;
  }
  if (InstructionIsUnavailable(instr)) {
    return memo[instr] = true;
  }
  for (const HloComputation* called : instr->called_computations()) {
    for (const HloInstruction* sub : called->instructions()) {
      if (ContainsUnavailableInstruction(sub, memo)) {
        return memo[instr] = true;
      }
    }
  }
  return memo[instr] = false;
}

bool IsControlFlow(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kWhile ||
         instr->opcode() == HloOpcode::kConditional;
}

// A maximal schedule-contiguous run of region-eligible instructions.
struct Region {
  std::vector<HloInstruction*> members;  // in topological order
};

// Enqueues the control-flow body/condition/branch computations directly
// reachable from `comp`'s current instructions (while body+condition,
// conditional branches). Only called on computations we will partition, after
// they have been partitioned, so a control-flow op that was swallowed into a
// hoisted `_region` no longer appears here — its body is therefore not
// re-partitioned (it is "the called computation of an already-hoisted
// region"). Fusion computations and generic call/map/reduce to_apply bodies
// are never enqueued: only control-flow bodies, whose per-op dispatch cost
// scales with trip count.
// Worklist entries carry whether the computation executes in a dispatch-
// amplified context: inside a while loop (directly, or transitively through
// nested control flow), every thunk dispatch is paid once per iteration.
void EnqueueControlFlowBodies(
    HloComputation* comp, bool comp_amplified,
    std::vector<std::pair<HloComputation*, bool>>& worklist,
    absl::flat_hash_set<HloComputation*>& seen) {
  for (HloInstruction* instr : comp->instructions()) {
    auto enqueue = [&](HloComputation* c, bool amplified) {
      if (seen.insert(c).second) {
        worklist.push_back({c, amplified});
      }
    };
    if (instr->opcode() == HloOpcode::kWhile) {
      enqueue(instr->while_body(), /*amplified=*/true);
      enqueue(instr->while_condition(), /*amplified=*/true);
    } else if (instr->opcode() == HloOpcode::kConditional) {
      for (HloComputation* branch : instr->branch_computations()) {
        enqueue(branch, comp_amplified);
      }
    }
  }
}

}  // namespace

SmallRegionHoistingPass::SmallRegionHoistingPass(
    int64_t small_buffer_access_size, int64_t min_region_size,
    bool exclude_nonscalar_constants)
    : small_buffer_access_size_(small_buffer_access_size),
      min_region_size_(min_region_size),
      exclude_nonscalar_constants_(exclude_nonscalar_constants) {}

// Partitions `comp`'s schedule into maximal region-eligible runs and outlines
// each cost-model-passing run into an `xla_cpu_small_call` kCall. Returns true
// if `comp` was modified. `unavailable_memo` is shared across computations.
// True if `instr` is a non-scalar constant that, under the region flag, must
// be excluded from region membership so it becomes a region input (kCall
// operand) rather than an unreachable region-internal value.
static bool IsExcludedNonscalarConstant(const HloInstruction* instr,
                                        bool exclude_nonscalar_constants) {
  return exclude_nonscalar_constants &&
         instr->opcode() == HloOpcode::kConstant &&
         !ShapeUtil::IsEffectiveScalar(instr->shape());
}

static absl::StatusOr<bool> PartitionComputation(
    HloModule* module, HloComputation* comp, bool amplified,
    int64_t small_buffer_access_size, int64_t min_region_size,
    bool exclude_nonscalar_constants,
    absl::flat_hash_map<const HloInstruction*, bool>& unavailable_memo) {
  // Phase 1: partition the computation's topological order into maximal runs
  // of region-eligible instructions, split at unavailable instructions, with
  // per-member byte footprints. Parameters are region inputs, not members,
  // and do not break a run.
  struct Run {
    std::vector<HloInstruction*> members;
    std::vector<int64_t> bytes;
    int64_t total_bytes = 0;
  };
  std::vector<Run> runs;
  Run current;
  auto close_run = [&]() {
    if (!current.members.empty()) {
      runs.push_back(std::move(current));
      current = Run{};
    }
  };
  HloCostAnalysis cost_analysis(&CpuExecutable::ShapeSizeBytes);
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      continue;
    }
    // Under the region flag, non-scalar constants are not region members:
    // leave them in the parent so they become region inputs (kCall operands),
    // reachable by the tiled region kernel ABI. They don't break a run, just
    // like parameters.
    if (IsExcludedNonscalarConstant(instr, exclude_nonscalar_constants)) {
      continue;
    }
    if (ContainsUnavailableInstruction(instr, unavailable_memo)) {
      close_run();
      continue;
    }
    RETURN_IF_ERROR(cost_analysis.RevisitInstruction(instr));
    current.members.push_back(instr);
    current.bytes.push_back(cost_analysis.bytes_accessed(*instr));
    current.total_bytes += current.bytes.back();
  }
  close_run();

  // Phase 2: apply the byte budget. A run that fits becomes one region, as
  // before. An over-budget run is greedily segmented at oversized members and
  // at budget overflow — but a segment is only KEPT where a loop amplifies
  // its dispatch savings: in a while-amplified computation, or when the
  // segment itself contains control flow. A once-per-call straight-line
  // segment saves only nanoseconds of dispatch while serializing thunks the
  // executor could have overlapped (measured -13% on the Gemma3 1B benchmark
  // when such segments were folded), so those runs stay all-or-nothing.
  // Segmentation is what rescues a cheap million-iteration loop whose
  // neighbors are large loop-invariant fusions (jax-ml/jax discussion
  // #24501): the loop's segment survives the cut around the big neighbors.
  std::vector<Region> regions;
  for (Run& run : runs) {
    if (run.total_bytes < small_buffer_access_size) {
      regions.push_back(Region{std::move(run.members)});
      continue;
    }
    Region segment;
    int64_t segment_bytes = 0;
    auto flush_segment = [&]() {
      if (!segment.members.empty() &&
          (amplified || absl::c_any_of(segment.members, IsControlFlow))) {
        regions.push_back(std::move(segment));
      }
      segment = Region{};
      segment_bytes = 0;
    };
    for (size_t i = 0; i < run.members.size(); ++i) {
      const int64_t member_bytes = run.bytes[i];
      if (member_bytes >= small_buffer_access_size) {
        flush_segment();
        continue;
      }
      if (segment_bytes + member_bytes >= small_buffer_access_size) {
        flush_segment();
      }
      segment.members.push_back(run.members[i]);
      segment_bytes += member_bytes;
    }
    flush_segment();
  }

  bool changed = false;
  for (const Region& region : regions) {
    absl::flat_hash_set<const HloInstruction*> member_set(
        region.members.begin(), region.members.end());

    // Cost-model gate: a region is worth a single kernel when it is small and
    // either has enough instructions to beat per-op dispatch, or contains
    // control flow (whose dispatch cost scales with trip count regardless of
    // static instruction count).
    bool contains_control_flow =
        absl::c_any_of(region.members, IsControlFlow);
    if (region.members.size() <
            static_cast<size_t>(min_region_size) &&
        !contains_control_flow) {
      continue;
    }

    // The byte budget was already enforced during run construction above.

    // Conservative correctness guard: don't hoist a region whose members carry
    // control dependencies that CROSS the region boundary — outlining would
    // drop the ordering they encode. Side-effecting ops (infeed/outfeed/
    // custom-call/collectives) are already region boundaries, so such crossing
    // control edges are rare in practice; skipping these regions is safe and
    // costs only the status quo. Control deps internal to the region
    // (member<->member) are fine: the region collapses into one kernel and that
    // ordering is subsumed by data dependencies inside it.
    auto crosses_boundary = [&](const HloInstruction* other) {
      return !member_set.contains(other);
    };
    bool has_boundary_control_dep =
        absl::c_any_of(region.members, [&](const HloInstruction* member) {
          return absl::c_any_of(member->control_predecessors(),
                                crosses_boundary) ||
                 absl::c_any_of(member->control_successors(), crosses_boundary);
        });
    if (has_boundary_control_dep) {
      continue;
    }

    // Liveness boundary. Inputs: values used by the region but defined outside
    // it (computation parameters or earlier instructions). Outputs: region
    // members used outside the region, or the computation root.
    std::vector<HloInstruction*> inputs;
    absl::flat_hash_set<const HloInstruction*> input_set;
    for (HloInstruction* member : region.members) {
      for (HloInstruction* operand : member->operands()) {
        if (!member_set.contains(operand) && input_set.insert(operand).second) {
          inputs.push_back(operand);
        }
      }
    }
    std::vector<HloInstruction*> outputs;
    for (HloInstruction* member : region.members) {
      bool used_outside = member == comp->root_instruction() ||
                          absl::c_any_of(member->users(),
                                         [&](const HloInstruction* user) {
                                           return !member_set.contains(user);
                                         });
      if (used_outside) {
        outputs.push_back(member);
      }
    }
    // A region whose only effect is its root must have at least one output.
    if (outputs.empty()) {
      continue;
    }

    // Build the outlined computation: a parameter per input, a clone per
    // member, rooted at the single output or a tuple of outputs.
    HloComputation::Builder builder(
        absl::StrCat(region.members.back()->name(), "_region"));
    absl::flat_hash_map<const HloInstruction*, HloInstruction*> map;
    for (int64_t i = 0; i < inputs.size(); ++i) {
      map[inputs[i]] = builder.AddInstruction(HloInstruction::CreateParameter(
          i, inputs[i]->shape(), absl::StrCat("p", i)));
    }
    for (HloInstruction* member : region.members) {
      std::vector<HloInstruction*> new_operands;
      new_operands.reserve(member->operand_count());
      for (HloInstruction* operand : member->operands()) {
        new_operands.push_back(map.at(operand));
      }
      map[member] = builder.AddInstruction(
          member->CloneWithNewOperands(member->shape(), new_operands));
    }
    HloInstruction* outlined_root;
    Shape call_shape;
    if (outputs.size() == 1) {
      outlined_root = map.at(outputs[0]);
      call_shape = outputs[0]->shape();
    } else {
      std::vector<HloInstruction*> output_clones;
      output_clones.reserve(outputs.size());
      std::vector<Shape> output_shapes;
      output_shapes.reserve(outputs.size());
      for (HloInstruction* output : outputs) {
        output_clones.push_back(map.at(output));
        output_shapes.push_back(output->shape());
      }
      outlined_root =
          builder.AddInstruction(HloInstruction::CreateTuple(output_clones));
      call_shape = ShapeUtil::MakeTupleShape(output_shapes);
    }
    HloComputation* outlined =
        module->AddEmbeddedComputation(builder.Build(outlined_root));

    HloInstruction* call = comp->AddInstruction(
        HloInstruction::CreateCall(call_shape, inputs, outlined));
    call->add_frontend_attribute("xla_cpu_small_call", "true");

    // Redirect external uses to the call (or per-output get-tuple-elements).
    if (outputs.size() == 1) {
      RETURN_IF_ERROR(outputs[0]->ReplaceAllUsesWith(call));
    } else {
      for (int64_t i = 0; i < outputs.size(); ++i) {
        HloInstruction* gte =
            comp->AddInstruction(HloInstruction::CreateGetTupleElement(
                outputs[i]->shape(), call, i));
        RETURN_IF_ERROR(outputs[i]->ReplaceAllUsesWith(gte));
      }
    }

    // Remove the now-dead members in reverse topological order.
    for (auto it = region.members.rbegin(); it != region.members.rend();
         ++it) {
      RETURN_IF_ERROR((*it)->SafelyDropAllControlDependencies());
      RETURN_IF_ERROR(comp->RemoveInstruction(*it));
    }
    changed = true;
  }

  return changed;
}

absl::StatusOr<bool> SmallRegionHoistingPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry = module->entry_computation();
  if (entry == nullptr) {
    return false;
  }

  // Process the entry, then descend into control-flow bodies. We enqueue a
  // body only AFTER partitioning the computation that contains its
  // while/conditional op: if that op was swallowed into a hoisted `_region`,
  // it is gone from the computation and its body is not enqueued (we do not
  // partition the called computations of an already-hoisted region). The
  // `_region` computations we create are never enqueued because we only
  // descend through control-flow ops, never through the small_call kCall.
  std::vector<std::pair<HloComputation*, bool>> worklist = {
      {entry, /*amplified=*/false}};
  absl::flat_hash_set<HloComputation*> seen = {entry};

  absl::flat_hash_map<const HloInstruction*, bool> unavailable_memo;
  bool changed = false;
  for (size_t i = 0; i < worklist.size(); ++i) {
    auto [comp, amplified] = worklist[i];
    ASSIGN_OR_RETURN(
        bool comp_changed,
        PartitionComputation(module, comp, amplified,
                             small_buffer_access_size_, min_region_size_,
                             exclude_nonscalar_constants_, unavailable_memo));
    changed |= comp_changed;
    EnqueueControlFlowBodies(comp, amplified, worklist, seen);
  }
  return changed;
}

}  // namespace xla::cpu
