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

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/logical_buffer.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

class RegionKernelEmitterTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        [](const BufferValue& buffer) {
          return CpuExecutable::ShapeSizeBytes(buffer.shape());
        },
        &alias_info_, [](LogicalBuffer::Color) { return /*alignment=*/1; },
        BufferAssigner::Options{});
  }

  AliasInfo alias_info_;
};

// A straight-line, single-output-shape elementwise region.
static constexpr absl::string_view kStraightLineRegionHlo = R"(
  HloModule region_module

  %add_chain_region (p0: f32[256], p1: f32[256]) -> f32[256] {
    %p0 = f32[256] parameter(0)
    %p1 = f32[256] parameter(1)
    %add = f32[256] add(%p0, %p1)
    %mul = f32[256] multiply(%add, %p0)
    ROOT %sub = f32[256] subtract(%mul, %p1)
  }

  ENTRY %main (a: f32[256], b: f32[256]) -> f32[256] {
    %a = f32[256] parameter(0)
    %b = f32[256] parameter(1)
    ROOT %call = f32[256] call(%a, %b), to_apply=%add_chain_region,
      frontend_attributes={xla_cpu_small_call="true"}
  }
)";

// A multi-shape region: dot -> reduce -> divide -> broadcast -> subtract, the
// frag shape. Every step changes the indexing domain, so this exercises the
// xtile tensor-threading path (Increment 1').
static constexpr absl::string_view kMultiShapeRegionHlo = R"(
  HloModule region_module

  %add_reducer (a: f32[], b: f32[]) -> f32[] {
    %a = f32[] parameter(0)
    %b = f32[] parameter(1)
    ROOT %add = f32[] add(%a, %b)
  }

  %frag_region (p0: f32[16,16], p1: f32[16]) -> f32[16] {
    %p0 = f32[16,16] parameter(0)
    %p1 = f32[16] parameter(1)
    %dot = f32[16] dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    %tanh = f32[16] tanh(%dot)
    %zero = f32[] constant(0)
    %reduce = f32[] reduce(%tanh, %zero), dimensions={0}, to_apply=%add_reducer
    %n = f32[] constant(16)
    %div = f32[] divide(%reduce, %n)
    %bcast = f32[16] broadcast(%div), dimensions={}
    ROOT %sub = f32[16] subtract(%tanh, %bcast)
  }

  ENTRY %main (a: f32[16,16], b: f32[16]) -> f32[16] {
    %a = f32[16,16] parameter(0)
    %b = f32[16] parameter(1)
    ROOT %call = f32[16] call(%a, %b), to_apply=%frag_region,
      frontend_attributes={xla_cpu_small_call="true"}
  }
)";

// A reduce -> broadcast (dot-free) multi-shape region.
static constexpr absl::string_view kReduceBroadcastRegionHlo = R"(
  HloModule region_module

  %add_reducer (a: f32[], b: f32[]) -> f32[] {
    %a = f32[] parameter(0)
    %b = f32[] parameter(1)
    ROOT %add = f32[] add(%a, %b)
  }

  %rb_region (p0: f32[16]) -> f32[16] {
    %p0 = f32[16] parameter(0)
    %zero = f32[] constant(0)
    %reduce = f32[] reduce(%p0, %zero), dimensions={0}, to_apply=%add_reducer
    %bcast = f32[16] broadcast(%reduce), dimensions={}
    ROOT %sub = f32[16] subtract(%p0, %bcast)
  }

  ENTRY %main (a: f32[16]) -> f32[16] {
    %a = f32[16] parameter(0)
    ROOT %call = f32[16] call(%a), to_apply=%rb_region,
      frontend_attributes={xla_cpu_small_call="true"}
  }
)";

TEST_F(RegionKernelEmitterTest, StraightLineRegionIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kStraightLineRegionHlo));
  auto* call = hlo_module->entry_computation()->root_instruction();
  EXPECT_TRUE(RegionKernelEmitter::IsSupportedRegion(call));
}

TEST_F(RegionKernelEmitterTest, MultiShapeRegionIsSupported) {
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kMultiShapeRegionHlo));
  auto* call = hlo_module->entry_computation()->root_instruction();
  // Multi-shape (dot/reduce/broadcast) is now in the envelope: it routes
  // through the xtile tiled emitter.
  EXPECT_TRUE(RegionKernelEmitter::IsSupportedRegion(call));
}

// RED -> GREEN: a multi-shape straight-line region emits via the tiled path as
// a single tiled kernel with `tiled_emitter` provenance.
TEST_F(RegionKernelEmitterTest, MultiShapeRegionEmitsViaTiledPath) {
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kMultiShapeRegionHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_use_fusion_emitters(true);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto* call = hlo_module->entry_computation()->root_instruction();

  auto mlir_context = FusionCompiler::CreateContext();
  RegionKernelEmitter emitter(call, buffer_assignment.get(),
                              mlir_context.get());
  TF_ASSERT_OK_AND_ASSIGN(auto kernel_definition,
                          emitter.EmitKernelDefinition());

  std::string mlir_string =
      kernel_definition.source().ToString();
  // One tiled kernel: the module carries the tiled_emitter provenance tag and
  // the xtile entry func.
  TF_ASSERT_OK_AND_ASSIGN(
      bool matched,
      RunFileCheck(mlir_string, R"(
        CHECK: tiled_emitter
        CHECK: xtile.entry
      )"));
  EXPECT_TRUE(matched);
}

TEST_F(RegionKernelEmitterTest, ReduceBroadcastRegionEmitsViaTiledPath) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto hlo_module, ParseAndReturnVerifiedModule(kReduceBroadcastRegionHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_use_fusion_emitters(true);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto* call = hlo_module->entry_computation()->root_instruction();

  auto mlir_context = FusionCompiler::CreateContext();
  RegionKernelEmitter emitter(call, buffer_assignment.get(),
                              mlir_context.get());
  TF_ASSERT_OK_AND_ASSIGN(auto kernel_definition,
                          emitter.EmitKernelDefinition());
  std::string mlir_string = kernel_definition.source().ToString();
  TF_ASSERT_OK_AND_ASSIGN(bool matched,
                          RunFileCheck(mlir_string, "CHECK: tiled_emitter"));
  EXPECT_TRUE(matched);
}

// A region whose body already contains NESTED kLoop fusions (the live,
// post-instruction-fusion shape). Operands are parameters only -- NO
// materialized non-scalar constants -- so this isolates the nested-fusion
// blocker (Step 1). Before flattening, the tiled emitter rejects the kFusion
// members; after Defuse()-flattening it must emit as ONE tiled kernel.
static constexpr absl::string_view kNestedFusionRegionHlo = R"(
  HloModule region_module

  %inner_add (x: f32[16], y: f32[16]) -> f32[16] {
    %x = f32[16] parameter(0)
    %y = f32[16] parameter(1)
    %add = f32[16] add(%x, %y)
    ROOT %mul = f32[16] multiply(%add, %x)
  }

  %inner_neg (z: f32[16]) -> f32[16] {
    %z = f32[16] parameter(0)
    %neg = f32[16] negate(%z)
    ROOT %abs = f32[16] abs(%neg)
  }

  %nested_region (p0: f32[16], p1: f32[16]) -> f32[16] {
    %p0 = f32[16] parameter(0)
    %p1 = f32[16] parameter(1)
    %f0 = f32[16] fusion(%p0, %p1), kind=kLoop, calls=%inner_add
    %f1 = f32[16] fusion(%f0), kind=kLoop, calls=%inner_neg
    ROOT %sub = f32[16] subtract(%f1, %p0)
  }

  ENTRY %main (a: f32[16], b: f32[16]) -> f32[16] {
    %a = f32[16] parameter(0)
    %b = f32[16] parameter(1)
    ROOT %call = f32[16] call(%a, %b), to_apply=%nested_region,
      frontend_attributes={xla_cpu_small_call="true"}
  }
)";

// RED -> GREEN: a region whose live body contains nested kLoop fusions emits
// as ONE tiled kernel. Before Step 1 (Defuse flattening) the tiled emitter
// rejects the kFusion members; after flattening the view has no nested fusions
// and routes through the tiled path.
TEST_F(RegionKernelEmitterTest, NestedFusionRegionEmitsViaTiledPath) {
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kNestedFusionRegionHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_use_fusion_emitters(true);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto* call = hlo_module->entry_computation()->root_instruction();

  auto mlir_context = FusionCompiler::CreateContext();
  RegionKernelEmitter emitter(call, buffer_assignment.get(),
                              mlir_context.get());
  TF_ASSERT_OK_AND_ASSIGN(auto kernel_definition,
                          emitter.EmitKernelDefinition());
  std::string mlir_string = kernel_definition.source().ToString();
  TF_ASSERT_OK_AND_ASSIGN(
      bool matched,
      RunFileCheck(mlir_string, R"(
        CHECK: tiled_emitter
        CHECK: xtile.entry
      )"));
  EXPECT_TRUE(matched);
}

TEST_F(RegionKernelEmitterTest, ScatterRegionIsDeclined) {
  static constexpr absl::string_view kScatterRegionHlo = R"(
    HloModule region_module

    %update (a: f32[], b: f32[]) -> f32[] {
      %a = f32[] parameter(0)
      ROOT %b = f32[] parameter(1)
    }

    %scatter_region (p0: f32[8], p1: s32[1,1], p2: f32[1]) -> f32[8] {
      %p0 = f32[8] parameter(0)
      %p1 = s32[1,1] parameter(1)
      %p2 = f32[1] parameter(2)
      ROOT %scatter = f32[8] scatter(%p0, %p1, %p2),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1,
        to_apply=%update
    }

    ENTRY %main (a: f32[8], b: s32[1,1], c: f32[1]) -> f32[8] {
      %a = f32[8] parameter(0)
      %b = s32[1,1] parameter(1)
      %c = f32[1] parameter(2)
      ROOT %call = f32[8] call(%a, %b, %c), to_apply=%scatter_region,
        frontend_attributes={xla_cpu_small_call="true"}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kScatterRegionHlo));
  auto* call = hlo_module->entry_computation()->root_instruction();
  // Scatter is Increment 2: declined here, falls back to legacy.
  EXPECT_FALSE(RegionKernelEmitter::IsSupportedRegion(call));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
