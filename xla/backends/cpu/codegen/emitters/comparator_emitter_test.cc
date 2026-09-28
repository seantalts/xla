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

#include "xla/backends/cpu/codegen/emitters/comparator_emitter.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/testlib/kernel_runner.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/transforms/expanders/comparison_expander.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace xla::cpu {
namespace {

using ::testing::HasSubstr;

using Comparator = bool(const void** data);
using Arg = std::variant<bool, int32_t, uint32_t, float, bfloat16>;

struct Call {
  std::vector<Arg> args;
  bool expected;
};

template <typename... Args>
Call True(Args... args) {
  return {{args...}, true};
}
template <typename... Args>
Call False(Args... args) {
  return {{args...}, false};
}

struct Case {
  Case(std::string name, std::string hlo, std::vector<Call> calls)
      : name(std::move(name)), hlo(std::move(hlo)), calls(std::move(calls)) {}
  std::string name;
  std::string hlo;
  std::vector<Call> calls;
};

struct JittedComparator {
  std::unique_ptr<FunctionLibrary> library;
  Comparator* fn = nullptr;
  std::string llvm_ir;
};

absl::StatusOr<JittedComparator> Jit(absl::string_view hlo,
                                     absl::string_view name) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                        ParseAndReturnUnverifiedModule(hlo));
  ABSL_RETURN_IF_ERROR(ComparisonExpander().Run(hlo_module.get()).status());
  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  const HloComputation* comparator =
      hlo_module->entry_computation()->root_instruction()->to_apply();
  ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                        EmitComparatorModule(*mlir_context, *comparator, name));
  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{/*vector_width=*/256,
                                                  /*verification_level=*/1,
                                                  /*fast_min_max=*/false});
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<llvm::Module> llvm_module,
                        compiler.Compile(*llvm_context, *mlir_module));

  JittedComparator result;
  result.llvm_ir = llvm_ir::DumpToString(llvm_module.get());
  ABSL_ASSIGN_OR_RETURN(JitCompiler jit,
                        KernelRunner::CreateJitCompiler(hlo_module->config()));
  ABSL_RETURN_IF_ERROR(jit.AddModule(llvm::orc::ThreadSafeModule(
      std::move(llvm_module), std::move(llvm_context))));
  ABSL_ASSIGN_OR_RETURN(
      result.library,
      std::move(jit).Compile({FunctionLibrary::Sym<Comparator>(name)}));
  ABSL_ASSIGN_OR_RETURN(result.fn,
                        result.library->ResolveFunction<Comparator>(name));
  return result;
}

std::string Compare(absl::string_view type, absl::string_view direction) {
  return absl::StrCat("cmp {\n  p0 = ", type, "[] parameter(0)\n  p1 = ", type,
                      "[] parameter(1)\n  ROOT c = pred[] compare(p0, p1), "
                      "direction=",
                      direction, "\n}\nENTRY e {\n  x = ", type,
                      "[8] parameter(0)\n  ROOT s = ", type,
                      "[8] sort(x), dimensions={0}, to_apply=cmp\n}");
}

constexpr absl::string_view kBf16TotalOrder = R"(
  region_0.1 {
    sort.2 = bf16[] parameter(0)
    ne.2 = pred[] compare(sort.2, sort.2), direction=NE
    constant.2 = bf16[] constant(nan)
    constant.3 = bf16[] constant(0)
    eq.2 = pred[] compare(sort.2, constant.3), direction=EQ
    select_n.4 = bf16[] select(eq.2, constant.3, sort.2)
    select_n.5 = bf16[] select(ne.2, constant.2, select_n.4)
    sort.3 = bf16[] parameter(1)
    ne.3 = pred[] compare(sort.3, sort.3), direction=NE
    eq.3 = pred[] compare(sort.3, constant.3), direction=EQ
    select_n.6 = bf16[] select(eq.3, constant.3, sort.3)
    select_n.7 = bf16[] select(ne.3, constant.2, select_n.6)
    ROOT lt = pred[] compare(select_n.5, select_n.7), direction=LT, type=TOTALORDER
  }
  ENTRY e {
    x = bf16[8] parameter(0)
    ROOT s = bf16[8] sort(x), dimensions={0}, is_stable=true, to_apply=region_0.1
  })";

constexpr absl::string_view kArgsortKeyValue = R"(
  lt {
    p0 = f32[] parameter(0)
    p1 = f32[] parameter(1)
    p2 = s32[] parameter(2)
    p3 = s32[] parameter(3)
    ROOT cmp = pred[] compare(p0, p1), direction=LT, type=TOTALORDER
  }
  ENTRY e {
    k = f32[8] parameter(0)
    v = s32[8] parameter(1)
    ROOT s = (f32[8], s32[8]) sort(k, v), dimensions={0}, to_apply=lt
  })";

std::vector<Case> Cases() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const bfloat16 bnan = std::numeric_limits<bfloat16>::quiet_NaN();
  const bfloat16 b0(0.0f), bneg0(-0.0f), b1(1.0f), b2(2.0f);
  return {
      Case("f32_lt", Compare("f32", "LT"),
           {True(1.0f, 2.0f), False(2.0f, 1.0f), False(1.0f, 1.0f),
            True(-1.0f, 0.0f)}),
      Case("bf16_totalorder", std::string(kBf16TotalOrder),
           {True(b1, b2), False(b2, b1), False(bneg0, b0), False(b0, bneg0),
            True(bneg0, b1), True(b2, bnan), False(bnan, b2), False(bnan, bnan),
            False(-bnan, bnan), True(b2, -bnan)}),
      Case("s32_gt", Compare("s32", "GT"),
           {True(5, 3), False(3, 5), False(3, 3), False(-1, 1), True(1, -1)}),
      Case("u32_lt", Compare("u32", "LT"),
           {True(1u, 0xFFFFFFFFu), False(0xFFFFFFFFu, 1u), False(7u, 7u)}),
      Case("pred_lt", Compare("pred", "LT"),
           {True(false, true), False(true, false), False(true, true),
            False(false, false)}),
      Case("argsort_key_value", std::string(kArgsortKeyValue),
           {True(1.0f, 2.0f, 9, 3), False(2.0f, 1.0f, 3, 9),
            True(1.0f, nan, 0, 0), False(nan, 1.0f, 0, 0),
            True(-0.0f, 0.0f, 0, 0), False(0.0f, -0.0f, 0, 0)}),
  };
}

class ComparatorEmitterTest : public ::testing::TestWithParam<Case> {};

TEST_P(ComparatorEmitterTest, Compares) {
  const Case& c = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(JittedComparator jitted, Jit(c.hlo, c.name));
  EXPECT_THAT(jitted.llvm_ir,
              HasSubstr(absl::StrCat("zeroext i1 @", c.name, "(ptr")));
  for (const Call& call : c.calls) {
    std::vector<const void*> data;
    for (const Arg& arg : call.args) {
      data.push_back(std::visit(
          [](const auto& value) -> const void* { return &value; }, arg));
    }
    EXPECT_EQ(jitted.fn(data.data()), call.expected) << c.name;
  }
}

INSTANTIATE_TEST_SUITE_P(ComparatorEmitter, ComparatorEmitterTest,
                         ::testing::ValuesIn(Cases()),
                         [](const auto& info) { return info.param.name; });

TEST(ComparatorEmitterTest, RejectsNonScalarParameters) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnUnverifiedModule(Compare("f32", "LT")));
  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  EXPECT_THAT(
      EmitComparatorModule(*mlir_context, *hlo_module->entry_computation(), "e")
          .status()
          .message(),
      HasSubstr("must be a scalar"));
}

}  // namespace
}  // namespace xla::cpu
