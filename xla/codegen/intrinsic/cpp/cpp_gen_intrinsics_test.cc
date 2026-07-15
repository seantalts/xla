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

#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"

#include <array>
#include <cmath>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_32_ll.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"

namespace xla::codegen {
namespace {

using ::xla::codegen::intrinsic::JitRunner;

// The MLIR intrinsic lowering always calls the canonical intrinsic with a
// direct signature `<N x T> @xla.sin.vNfT(<N x T>)`. On arm64-apple the bitcode
// compiles wide (>= 256-bit) vector returns indirectly as
// `void @sym(ptr sret(<N x T>), ptr)`. GetCppGenFunction must hand back a
// function with the *direct* signature (via a synthesized adapter) so the call
// the JIT emits binds to a matching body, or JIT execution corrupts the stack.
TEST(CppGenIntrinsicsTest, SinV8F32HasDirectSignature) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary32LlIr);

  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.sin.v8f32");
  ASSERT_NE(fn, nullptr);

  llvm::Type* v8f32 = llvm::VectorType::get(
      llvm::Type::getFloatTy(context), llvm::ElementCount::getFixed(8));
  // Expected direct type: <8 x float>(<8 x float>).
  EXPECT_EQ(fn->getReturnType(), v8f32)
      << "return type is not <8 x float> (indirect sret ABI leaked through)";
  ASSERT_EQ(fn->arg_size(), 1u);
  EXPECT_EQ(fn->getArg(0)->getType(), v8f32)
      << "argument type is not <8 x float>";
}

// Idempotency: calling GetCppGenFunction repeatedly for the same symbol must
// return the same direct-signature adapter and must not re-wrap it.
TEST(CppGenIntrinsicsTest, SinV8F32AdapterIsIdempotent) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary32LlIr);

  llvm::Function* first = GetCppGenFunction(module.get(), "xla.sin.v8f32");
  llvm::Function* second = GetCppGenFunction(module.get(), "xla.sin.v8f32");
  EXPECT_EQ(first, second);

  llvm::Type* v8f32 = llvm::VectorType::get(
      llvm::Type::getFloatTy(context), llvm::ElementCount::getFixed(8));
  EXPECT_EQ(second->getReturnType(), v8f32);
}

// Scalar and 128-bit vector already have the direct ABI; the fast path must
// leave them untouched (no adapter, direct signature preserved).
TEST(CppGenIntrinsicsTest, SinScalarAndV4F32Untouched) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary32LlIr);

  llvm::Function* scalar = GetCppGenFunction(module.get(), "xla.sin.f32");
  EXPECT_TRUE(scalar->getReturnType()->isFloatTy());
  ASSERT_EQ(scalar->arg_size(), 1u);
  EXPECT_TRUE(scalar->getArg(0)->getType()->isFloatTy());
  // No `.body` twin should have been created for an already-direct function.
  EXPECT_EQ(module->getFunction("xla.sin.f32.body"), nullptr);
  EXPECT_EQ(module->getFunction("\01xla.sin.f32.body"), nullptr);

  llvm::Function* v4 = GetCppGenFunction(module.get(), "xla.sin.v4f32");
  llvm::Type* v4f32 = llvm::VectorType::get(
      llvm::Type::getFloatTy(context), llvm::ElementCount::getFixed(4));
  EXPECT_EQ(v4->getReturnType(), v4f32);
}

// The adapter body must compute sin correctly through the JIT. This exercises
// the whole indirect-return path: the adapter's alloca/store/call/load wrapping
// the sret body, executed for real. Before the fix the JIT would bind the
// direct call to the sret body and corrupt the stack.
TEST(CppGenIntrinsicsTest, SinV8F32ComputesSinThroughJit) {
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, llvm_ir::kEigenUnary32LlIr);

  // The embedded bitcode is compiled by the host toolchain and bakes in
  // host-specific codegen attributes (Darwin's "probe-stack"="__chkstk_darwin"
  // aborts the JIT AArch64 backend; target-cpu/target-features pin the build
  // host). CppGenIntrinsicLibrary::LinkIntoModule strips these in the real
  // pipeline; do the same here so the JIT can lower the body.
  for (llvm::Function& f : *module) {
    f.removeFnAttr("probe-stack");
    f.removeFnAttr("target-cpu");
    f.removeFnAttr("target-features");
  }

  // Materialize the direct-signature adapter under xla.sin.v8f32.
  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.sin.v8f32");
  // JitRunner looks the wrapped function up by symbol name; make it visible.
  fn->setLinkage(llvm::Function::ExternalLinkage);

  JitRunner jit(std::move(module), std::move(context));
  auto sin = jit.GetVectorizedFn<8, float, float>(std::string(fn->getName()));

  std::array<float, 8> x = {0.0f, 0.5f, -1.5f, 3.0f,
                            0.4f, 2.0f, -4.0f, 5.5f};
  std::array<float, 8> y = sin(x);
  for (int i = 0; i < 8; ++i) {
    // Custom f32 sin is within ~1 ULP; a loose absolute bound suffices here to
    // confirm correct computation (not stack corruption).
    EXPECT_NEAR(y[i], std::sin(x[i]), 1e-5f) << "lane " << i << " x=" << x[i];
  }
}

}  // namespace
}  // namespace xla::codegen
