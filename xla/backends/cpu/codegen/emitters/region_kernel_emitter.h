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
#ifndef XLA_BACKENDS_CPU_CODEGEN_EMITTERS_REGION_KERNEL_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_EMITTERS_REGION_KERNEL_EMITTER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// Stage-1 Increment-1' region emitter.
//
// Emits the `to_apply()` of an `xla_cpu_small_call` kCall instruction as a
// single tiled MLIR kernel via the existing xtile tiled-fusion emitter, as the
// MLIR analog of the legacy `ComputationKernelEmitter`.
//
// Increment 1' handles STRAIGHT-LINE, single-array-output regions of arbitrary
// (multi-)shape: dot / reduce / broadcast / elementwise chains. The region is
// reified as a throwaway `HloFusionInstruction` "view" (a kLoop fusion whose
// fused computation is a clone of `to_apply()`) and fed to
// `EmitTiledFusionKernel`, which walks the tiled computation in schedule order
// and threads intermediate tensors between differently-shaped ops. Intermediates
// are materialized as `memref.alloca` by the bufferization pipeline.
//
// For anything outside that envelope -- control flow (kWhile/kConditional),
// scatter, tuple roots / multi-output regions, or any op the tiled emitter does
// not support -- `EmitKernelDefinition` returns a non-OK status so the caller
// falls back to the legacy `ComputationKernelEmitter`. Scatter and control flow
// are Increment 2/3 and are intentionally NOT attempted here.
class RegionKernelEmitter final : public KernelEmitter<MlirKernelSource> {
 public:
  RegionKernelEmitter(const HloInstruction* call,
                      const BufferAssignment* buffer_assignment,
                      mlir::MLIRContext* mlir_context);

  absl::string_view name() const final { return "region_kernel_emitter"; }
  absl::StatusOr<KernelDefinition> EmitKernelDefinition() final;

  // Returns true if `call`'s region is within the Increment-1' envelope (so this
  // emitter can handle it). When false the caller should use the legacy path.
  static bool IsSupportedRegion(const HloInstruction* call);

 private:
  // Builds a throwaway single-fusion module whose fusion instruction is a kLoop
  // fusion wrapping a clone of the region body. The returned fusion is owned by
  // `view_module` (populated as an out-param). The fusion's operands and root
  // shape match the kCall, so its buffer slices resolve against the original
  // `buffer_assignment` via the kCall.
  absl::StatusOr<const HloFusionInstruction*> BuildFusionView(
      std::unique_ptr<HloModule>& view_module) const;

  const HloInstruction* call_;
  const HloComputation* region_;
  const BufferAssignment* buffer_assignment_;
  mlir::MLIRContext* mlir_context_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_REGION_KERNEL_EMITTER_H_
