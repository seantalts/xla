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

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
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

}  // namespace
}  // namespace xla::cpu
