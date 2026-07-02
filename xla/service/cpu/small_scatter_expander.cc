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

#include "xla/service/cpu/small_scatter_expander.h"

#include <cstdint>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::cpu {
namespace {

// Accumulates the byte size of every array leaf of `shape` into `footprint`.
// Returns false if any leaf is dynamic (no static byte size).
bool AccumulateStaticByteSize(const Shape& shape, int64_t* footprint) {
  bool all_static = true;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        if (!subshape.IsArray()) {
          return;
        }
        if (!subshape.is_static()) {
          all_static = false;
          return;
        }
        *footprint += ShapeUtil::ByteSizeOf(subshape);
      });
  return all_static;
}

}  // namespace

bool SmallScatterExpander::InstructionMatchesPattern(HloInstruction* inst) {
  if (!ScatterExpander::InstructionMatchesPattern(inst)) {
    return false;
  }
  // Total byte footprint: all operands (scatter targets, indices, updates)
  // plus all results (tuple leaves for variadic scatters).
  int64_t footprint = 0;
  for (const HloInstruction* operand : inst->operands()) {
    if (!AccumulateStaticByteSize(operand->shape(), &footprint)) {
      return false;
    }
  }
  if (!AccumulateStaticByteSize(inst->shape(), &footprint)) {
    return false;
  }
  return footprint < small_buffer_access_size_;
}

}  // namespace xla::cpu
