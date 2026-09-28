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

#include "xla/service/cpu/cpu_aot_loader.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

SymbolProto MakeSymbol(SymbolProto::FunctionTypeId type_id, const char* name) {
  SymbolProto symbol;
  symbol.set_function_type_id(type_id);
  symbol.set_name(name);
  return symbol;
}

TEST(CpuAotLoaderTest, ResolvesKernelAndComparatorV2Symbols) {
  std::vector<SymbolProto> protos = {
      MakeSymbol(SymbolProto::KERNEL, "kernel"),
      MakeSymbol(SymbolProto::COMPARATOR_V2, "comparator")};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<FunctionLibrary::Symbol> symbols,
                          GetCompiledSymbolsFromProto(protos));
  ASSERT_EQ(symbols.size(), 2);
  EXPECT_EQ(symbols[0].type_id,
            FunctionLibrary::Sym<FunctionLibrary::Kernel>("kernel").type_id);
  EXPECT_EQ(symbols[0].name, "kernel");
  EXPECT_EQ(
      symbols[1].type_id,
      FunctionLibrary::Sym<FunctionLibrary::Comparator>("comparator").type_id);
  EXPECT_EQ(symbols[1].name, "comparator");
}

TEST(CpuAotLoaderTest, RejectsLegacyComparatorSymbols) {
  std::vector<SymbolProto> protos = {
      MakeSymbol(SymbolProto::COMPARATOR, "comparator")};

  EXPECT_THAT(GetCompiledSymbolsFromProto(protos),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("legacy sort comparator ABI")));
}

}  // namespace
}  // namespace xla::cpu
