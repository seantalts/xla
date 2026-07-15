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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_32_ll.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_64_ll.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::codegen {

const std::string& GetCppGenIrString(
    const intrinsics::IntrinsicOptions& options) {
  if (options.Contains("+avx512f") && (options.prefer_vector_width == 512 ||
                                       options.prefer_vector_width == 0)) {
    return ::llvm_ir::kEigenUnary64LlIr;
  }
  return ::llvm_ir::kEigenUnary32LlIr;
}

bool AreEigenIntrinsicsAvailable() {
  return !GetCppGenIrString(intrinsics::IntrinsicOptions()).empty();
}

namespace {

// Looks up a bitcode function by `name`, retrying with the Mach-O '\01' prefix
// that clang applies to explicit asm() labels (see note below). Returns nullptr
// if neither form exists.
llvm::Function* LookupWithMachOPrefix(llvm::Module* module,
                                      absl::string_view name) {
  llvm::Function* func =
      module->getFunction(llvm::StringRef(name.data(), name.size()));
  if (func == nullptr) {
    // On Mach-O targets clang prefixes explicit asm() labels (used to name the
    // intrinsics in eigen_unary.h) with '\01' to suppress the leading-underscore
    // mangling, so the bitcode compiled on macOS carries that prefix while the
    // lookup name does not. Retry with the prefix. (No-op on ELF, where the
    // names already match.)
    std::string prefixed = "\01";
    prefixed.append(name.data(), name.size());
    func = module->getFunction(prefixed);
  }
  return func;
}

// The MLIR intrinsic lowering (LowerIntrinsicPattern -> GetOrInsertDeclaration)
// always declares and calls the canonical intrinsic with a *direct* signature:
// `<N x T> @xla.sin.vNfT(<N x T>)` (scalar `T @xla.sin.fT(T)`). But the bitcode
// produced by cc_to_llvm_ir compiles wide ext-vector returns *indirectly* on
// targets whose C ABI cannot return them in registers (arm64-apple for
// 256/512-bit vectors, x86 at default SSE2, etc.): clang emits
// `void @sym(ptr sret(<N x T>), ptr %arg)`. The direct-ABI call the JIT emits
// then binds to an sret-ABI body -> stack corruption -> SIGSEGV.
//
// Returns true iff `func` has the indirect (sret) form and therefore needs an
// adapter with the expected direct signature.
bool IsIndirectSretReturn(const llvm::Function* func) {
  if (!func->getReturnType()->isVoidTy()) {
    return false;
  }
  return func->arg_size() >= 1 &&
         func->hasParamAttribute(0, llvm::Attribute::StructRet);
}

// Synthesizes an adapter with the direct signature `<N x T> @<name>(arg...)`
// that the MLIR lowering expects, wrapping the indirect-ABI `body_func`
// (`void @<name>.body(ptr sret(<N x T>), <arg params>)`). The adapter:
//   - allocates the sret return slot,
//   - materializes each non-sret parameter: a pointer parameter (the bitcode
//     passes the input vector via a pointer) gets an alloca + store of the
//     incoming by-value vector; a value parameter is forwarded directly,
//   - calls body_func, loads the return slot, and returns it.
// Both the adapter and body are InternalLinkage + AlwaysInline so that, after
// inlining and opt, the adapter and its allocas vanish entirely.
llvm::Function* CreateDirectAdapter(llvm::Module* module,
                                    llvm::Function* body_func,
                                    const std::string& adapter_name) {
  llvm::LLVMContext& ctx = module->getContext();
  llvm::Type* ret_type = body_func->getParamStructRetType(0);
  CHECK(ret_type != nullptr) << "sret parameter has no pointee type";

  // The direct signature takes one value parameter per non-sret parameter of
  // the body. For a pointer parameter that carries a vector (the common case),
  // the direct parameter is that vector type (== ret_type for these unary ops);
  // for a value parameter, it is the same type.
  std::vector<llvm::Type*> adapter_param_types;
  for (unsigned i = 1; i < body_func->arg_size(); ++i) {
    llvm::Argument* arg = body_func->getArg(i);
    if (arg->getType()->isPointerTy()) {
      // Pointer-carried argument: the direct ABI passes the value. These unary
      // intrinsics always carry the same vector type as the return.
      adapter_param_types.push_back(ret_type);
    } else {
      adapter_param_types.push_back(arg->getType());
    }
  }

  llvm::FunctionType* adapter_type =
      llvm::FunctionType::get(ret_type, adapter_param_types, /*isVarArg=*/false);
  llvm::Function* adapter = llvm::Function::Create(
      adapter_type, llvm::Function::InternalLinkage, adapter_name, module);
  adapter->addFnAttr(llvm::Attribute::AlwaysInline);

  llvm::BasicBlock* entry =
      llvm::BasicBlock::Create(ctx, "entry", adapter);
  llvm::IRBuilder<> builder(entry);

  llvm::AllocaInst* ret_slot = builder.CreateAlloca(ret_type);
  ret_slot->setAlignment(llvm::Align(16));

  std::vector<llvm::Value*> call_args;
  call_args.push_back(ret_slot);
  for (unsigned i = 1; i < body_func->arg_size(); ++i) {
    llvm::Argument* body_arg = body_func->getArg(i);
    llvm::Argument* adapter_arg = adapter->getArg(i - 1);
    if (body_arg->getType()->isPointerTy()) {
      llvm::AllocaInst* arg_slot =
          builder.CreateAlloca(adapter_arg->getType());
      arg_slot->setAlignment(llvm::Align(16));
      builder.CreateStore(adapter_arg, arg_slot);
      call_args.push_back(arg_slot);
    } else {
      call_args.push_back(adapter_arg);
    }
  }

  builder.CreateCall(body_func, call_args);
  llvm::Value* result = builder.CreateLoad(ret_type, ret_slot);
  builder.CreateRet(result);
  return adapter;
}

}  // namespace

llvm::Function* GetCppGenFunction(llvm::Module* module,
                                  absl::string_view name) {
  llvm::Function* func = LookupWithMachOPrefix(module, name);
  CHECK(func != nullptr)
      << "CppGen function '" << name
      << "' was not found in the module. Ensure the "
         "function name is correct and the library "
         "containing it was linked by IntrinsicFunctionLib.\n"
      << llvm_ir::DumpToString(module);

  // Fast path: already a definition with the expected direct signature
  // (scalars and narrow vectors that the host C ABI returns in registers).
  if (!func->isDeclaration() && !IsIndirectSretReturn(func)) {
    func->setLinkage(llvm::Function::InternalLinkage);
    func->addFnAttr(llvm::Attribute::AlwaysInline);
    return func;
  }

  // Declaration-only: nothing to adapt (the caller supplies the definition).
  if (func->isDeclaration()) {
    return func;
  }

  // Indirect (sret) ABI: the MLIR lowering will emit a direct-signature call to
  // `name`, which must not bind to the sret body. Rename the sret body to a
  // private `.body` symbol and synthesize a direct-signature adapter under the
  // canonical `name`.
  //
  // Idempotency: on any subsequent call for the same symbol the lookup above
  // returns the direct-signature *adapter* (a definition that is not sret), so
  // the fast path handles it and we never reach here twice for one symbol.
  const std::string canonical_name = func->getName().str();
  const std::string body_name = canonical_name + ".body";

  // Demote the sret body to the private `.body` name so the adapter can claim
  // the canonical name that the MLIR lowering looks up. setName preserves any
  // Mach-O '\01' prefix already present on canonical_name.
  func->setName(body_name);
  func->setLinkage(llvm::Function::InternalLinkage);
  func->addFnAttr(llvm::Attribute::AlwaysInline);

  return CreateDirectAdapter(module, func, canonical_name);
}

std::unique_ptr<llvm::Module> ParseEmbeddedBitcode(
    llvm::LLVMContext& context, const std::string& bitcode,
    absl::string_view source_name) {
  if (bitcode.empty()) {
    LOG_FIRST_N(INFO, 1)
        << "Empty bitcode string provided for " << source_name
        << ". Optimizations relying on this IR will be disabled.";
    return std::make_unique<llvm::Module>("empty", context);
  }

  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(bitcode.data(), bitcode.size()),
      llvm::StringRef(source_name.data(), source_name.size()),
      /*RequiresNullTerminator=*/false);
  std::unique_ptr<llvm::Module> module =
      llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);

  CHECK(module != nullptr) << "Failed to parse IR: "
                           << diagnostic.getMessage().str() << "\n"
                           << bitcode;
  return module;
}

// The default LLVM diagnostic handler uses llvm::errs(), which is not
// thread-safe.
static void DiagnosticHandler(const llvm::DiagnosticInfo* diag_info,
                              void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info->print(diagnostic_printer);

  if (diag_info->getSeverity() == llvm::DS_Error) {
    LOG(ERROR) << error_string;
  } else {
    VLOG(1) << error_string;
  }
}

void CppGenIntrinsicLibrary::LinkIntoModule(llvm::Module& dst_module) const {
  llvm::LLVMContext& context = dst_module.getContext();

  std::unique_ptr<llvm::Module> lib_module =
      ParseEmbeddedBitcode(context, ir_text_, source_name_);

  std::vector<std::string> lib_functions;
  for (const auto& func : *lib_module) {
    if (!func.isDeclaration()) {
      lib_functions.push_back(func.getName().str());
    }
  }

  const llvm::DataLayout& hostDataLayout = dst_module.getDataLayout();
  lib_module->setDataLayout(hostDataLayout);

  auto old_handler = context.getDiagnosticHandlerCallBack();
  void* old_handler_context = context.getDiagnosticContext();

  context.setDiagnosticHandlerCallBack(DiagnosticHandler, nullptr);

  // Using static Linker::linkModules based on previous success, but matching
  // logic
  if (llvm::Linker::linkModules(dst_module, std::move(lib_module))) {
    LOG(FATAL) << "LLVM Linker failed to link CppGen library.";
  }

  context.setDiagnosticHandlerCallBack(old_handler, old_handler_context);

  for (const auto& func : lib_functions) {
    llvm::Function* linked_func = dst_module.getFunction(func);
    if (linked_func && !linked_func->isDeclaration()) {
      linked_func->setLinkage(llvm::Function::InternalLinkage);
      linked_func->addFnAttr(llvm::Attribute::AlwaysInline);
      // The bitcode is compiled by the host toolchain and carries host-specific
      // codegen attributes baked into the function. On Darwin clang emits
      // "probe-stack"="__chkstk_darwin", which makes the JIT AArch64 backend
      // abort with "Unsupported stack probing method"; it also pins
      // target-cpu/target-features to the build host. Strip these so the JIT
      // TargetMachine governs codegen. These functions are AlwaysInline'd into
      // XLA kernels and inherit the caller's attributes.
      linked_func->removeFnAttr("probe-stack");
      linked_func->removeFnAttr("target-cpu");
      linked_func->removeFnAttr("target-features");
    }
  }
}

}  // namespace xla::codegen
