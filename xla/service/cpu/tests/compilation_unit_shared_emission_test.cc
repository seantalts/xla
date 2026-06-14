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

#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla::cpu {
namespace {

class CompilationUnitSharedEmissionTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts =
        HloPjRtInterpreterReferenceMixin::GetDebugOptionsForTest();
    opts.set_xla_cpu_compilation_unit_shared_emission(true);
    return opts;
  }
};

TEST_F(CompilationUnitSharedEmissionTest, SingleMarkedCallRunsCorrectly) {
  constexpr absl::string_view kHlo = R"(
HloModule m
unit {
  p0 = f32[64] parameter(0)
  t = f32[64] exponential(p0)
  ROOT a = f32[64] add(t, t)
}
ENTRY e {
  p = f32[64] parameter(0)
  c = f32[64] call(p), to_apply=unit, frontend_attributes={compilation_unit="unit0"}
  ROOT r = f32[64] negate(c)
}
)";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(CompilationUnitSharedEmissionTest, ManySitesShareOneEmission) {
  constexpr absl::string_view kHlo = R"(
HloModule m
unit {
  p0 = f32[64] parameter(0)
  t = f32[64] exponential(p0)
  ROOT a = f32[64] add(t, t)
}
ENTRY e {
  p = f32[64] parameter(0)
  c0 = f32[64] call(p), to_apply=unit, frontend_attributes={compilation_unit="unit0"}
  c1 = f32[64] call(c0), to_apply=unit, frontend_attributes={compilation_unit="unit0"}
  c2 = f32[64] call(c1), to_apply=unit, frontend_attributes={compilation_unit="unit0"}
  ROOT r = f32[64] negate(c2)
}
)";

  // Part 1 — numerics.
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{1e-5, 1e-5}));

  // Part 2 — structure: compile directly and inspect the thunks. The three
  // marked call sites must share a single emission, i.e. all three
  // RemappedCallThunks must point at the SAME nested ThunkSequence.
  CpuCompiler compiler;
  Compiler::CompileOptions compile_options;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_compilation_unit_shared_emission(true);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> optimized,
      compiler.RunHloPasses(std::move(module), /*stream_exec=*/nullptr,
                            compile_options));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler.RunBackend(std::move(optimized), /*stream_exec=*/nullptr,
                          compile_options));
  auto* cpu_exe = static_cast<CpuExecutable*>(executable.get());
  ASSERT_TRUE(cpu_exe->has_thunks());

  absl::flat_hash_set<const ThunkSequence*> shared;
  int sites = 0;
  for (const std::unique_ptr<Thunk>& thunk :
       cpu_exe->thunks().thunk_sequence()) {
    for (const auto& [name, seq] : thunk->nested_thunks()) {
      if (absl::StrContains(name, "-called_sequence")) {
        shared.insert(seq);
        ++sites;
      }
    }
  }
  EXPECT_EQ(sites, 3);
  EXPECT_EQ(shared.size(), 1);
}

TEST_F(CompilationUnitSharedEmissionTest, MarkedCallInsideWhileBody) {
  constexpr absl::string_view kHlo = R"(
HloModule m
unit {
  p0 = f32[8] parameter(0)
  t = f32[8] multiply(p0, p0)
  ROOT a = f32[8] add(t, t)
}
body {
  p = (s32[], f32[8]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  x = f32[8] get-tuple-element(p), index=1
  c = f32[8] call(x), to_apply=unit, frontend_attributes={compilation_unit="unit0"}
  one = s32[] constant(1)
  ni = s32[] add(i, one)
  ROOT t = (s32[], f32[8]) tuple(ni, c)
}
cond {
  p = (s32[], f32[8]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(5)
  ROOT lt = pred[] compare(i, n), direction=LT
}
ENTRY e {
  z = s32[] constant(0)
  x = f32[8] parameter(0)
  init = (s32[], f32[8]) tuple(z, x)
  ROOT w = (s32[], f32[8]) while(init), condition=cond, body=body
}
)";
  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{1e-4, 1e-4}));
}

}  // namespace
}  // namespace xla::cpu
