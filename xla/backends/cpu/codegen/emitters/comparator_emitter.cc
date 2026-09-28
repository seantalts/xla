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
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitComparatorModule(
    mlir::MLIRContext& context, const HloComputation& comparator,
    absl::string_view function_name) {
  if (comparator.num_parameters() == 0) {
    return InvalidArgument("Sort comparator %s has no parameters",
                           comparator.name());
  }
  for (const HloInstruction* param : comparator.parameter_instructions()) {
    const Shape& shape = param->shape();
    if (!shape.IsArray() || shape.dimensions().size() != 0) {
      return InvalidArgument(
          "Sort comparator %s parameter %s must be a scalar, got %s",
          comparator.name(), param->name(), shape.ToString());
    }
    if (primitive_util::IsComplexType(shape.element_type())) {
      return Unimplemented(
          "Sort comparator %s parameter %s has complex type %s",
          comparator.name(), param->name(), shape.ToString());
    }
  }
  const Shape& root_shape = comparator.root_instruction()->shape();
  if (!root_shape.IsArray() || root_shape.dimensions().size() != 0 ||
      root_shape.element_type() != PRED) {
    return InvalidArgument("Sort comparator %s must return pred[], got %s",
                           comparator.name(), root_shape.ToString());
  }

  mlir::OpBuilder builder(&context);
  mlir::Location loc = mlir::NameLoc::get(builder.getStringAttr(function_name));
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(loc, function_name);
  (*module)->setAttr(xla::CpuMemoryRegionNameAttr::name,
                     builder.getStringAttr(absl::StrCat(
                         "xla_cpu_emitter__comparator__hlo_opcode__",
                         HloOpcodeString(HloOpcode::kSort))));

  emitters::PartitionedComputations computations(&comparator, &context);
  auto subgraph_to_fn = computations.DeclareFunctions(*module);
  for (auto& [subgraph, fn] : subgraph_to_fn) {
    if (absl::string_view(fn.getName()) == function_name) {
      return InvalidArgument(
          "Sort comparator function name %s collides with subgraph %s",
          function_name, subgraph->name);
    }
    fn->setAttr("llvm.always_inline", builder.getUnitAttr());
  }
  auto call_targets = computations.CreateCallTargetProvider(subgraph_to_fn);
  for (const auto& comp : computations.partitioned_computations()) {
    for (const auto& subgraph : comp.subgraphs()) {
      if (auto it = subgraph_to_fn.find(&subgraph);
          it != subgraph_to_fn.end()) {
        ABSL_RETURN_IF_ERROR(emitters::SubgraphToMlirFunction(
            comp, subgraph, it->second, call_targets, &context));
      }
    }
  }
  mlir::func::FuncOp body = call_targets(comparator.root_instruction());

  mlir::ImplicitLocOpBuilder b(loc, &context);
  b.setInsertionPointToStart(module->getBody());

  mlir::Type ptr_type = mlir::LLVM::LLVMPointerType::get(&context);
  mlir::Type i1_type = b.getI1Type();
  auto wrapper = mlir::func::FuncOp::create(
      b, function_name, b.getFunctionType({ptr_type}, {i1_type}));
  wrapper->setAttr("xla.entry", b.getUnitAttr());
  wrapper->setAttr("xla.cpu.is_wrapped", b.getUnitAttr());
  wrapper.setResultAttr(0, mlir::LLVM::LLVMDialect::getZExtAttrName(),
                        b.getUnitAttr());

  mlir::Block* entry = wrapper.addEntryBlock();
  b.setInsertionPointToStart(entry);
  mlir::Value data = entry->getArgument(0);

  llvm::SmallVector<mlir::Value> args;
  args.reserve(comparator.num_parameters());
  for (int64_t k = 0; k < comparator.num_parameters(); ++k) {
    mlir::Type value_type = emitters::PrimitiveTypeToMlirType(
        comparator.parameter_instruction(k)->shape().element_type(), b);
    mlir::Value slot = mlir::LLVM::GEPOp::create(
        b, ptr_type, ptr_type, data,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(k)},
        mlir::LLVM::GEPNoWrapFlags::inbounds);
    mlir::Value value_ptr = mlir::LLVM::LoadOp::create(b, ptr_type, slot);
    args.push_back(mlir::LLVM::LoadOp::create(b, value_type, value_ptr));
  }

  mlir::Value result = mlir::func::CallOp::create(b, body, args).getResult(0);
  if (result.getType() != i1_type) {
    result = mlir::arith::TruncIOp::create(b, i1_type, result);
  }
  mlir::func::ReturnOp::create(b, result);

  if (mlir::failed(mlir::verify(*module))) {
    return Internal("Emitted sort comparator module %s failed verification",
                    function_name);
  }
  return module;
}

}  // namespace xla::cpu
