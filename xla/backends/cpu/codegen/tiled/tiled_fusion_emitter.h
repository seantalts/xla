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

#ifndef XLA_BACKENDS_CPU_CODEGEN_TILED_TILED_FUSION_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_TILED_TILED_FUSION_EMITTER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

bool IsSupportedTilingType(PrimitiveType type);

struct TiledEmissionResult {
  absl::StatusOr<KernelDefinition<MlirKernelSource>> kernel;
  bool tiling_succeeded = true;
};

TiledEmissionResult EmitTiledFusionKernel(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, absl::string_view name,
    int64_t num_work_groups);

// Stage-1 region-compilation entry point. Emits `fusion_view` (a throwaway
// kLoop fusion that wraps a clone of an `xla_cpu_small_call` region body) as a
// single tiled kernel. Buffer slices are resolved against `region_call` -- the
// original kCall -- since that is the instruction buffer assignment saw;
// `fusion_view`'s operands and result shape match it. The region allow-list
// (dot/reduce/broadcast) is enabled.
TiledEmissionResult EmitTiledRegionKernel(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion_view,
    const HloInstruction& region_call, const BufferAssignment* buffer_assignment,
    absl::string_view name, int64_t num_work_groups);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_TILED_TILED_FUSION_EMITTER_H_
