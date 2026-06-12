/* Copyright 2024 The OpenXLA Authors.

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

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/type_util.h"

// The order of these includes is important.
#define GET_ATTRDEF_CLASSES
#include "xla/codegen/emitters/ir/xla_attrs.cc.inc"
#include "xla/codegen/emitters/ir/xla_enums.cc.inc"

namespace xla {
namespace {

struct XlaInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // Returns true if the given operation 'callable', that implements the
  // 'CallableOpInterface', can be inlined into the position given call
  // operation 'call', that is registered to the current dialect and implements
  // the `CallOpInterface`. 'wouldBeCloned' is set to true if the region of the
  // given 'callable' is set to be cloned during the inlining process, or false
  // if the region is set to be moved in-place (i.e. no duplicates would be
  // created).
  bool isLegalToInline(mlir::Operation* call, mlir::Operation* callable,
                       bool wouldBeCloned) const final {
    if (call->hasAttr("noinline")) return false;
    if (callable->hasAttr(emitters::kHasNoCompute)) return true;
    auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(callable);
    if (!func_op) {
      return false;
    }
    if (!mlir::isa<PureCallOp>(call)) {
      return false;
    }
    if (!func_op.getCallableRegion()) {
      return false;
    }
    // Inlining a pure call is always semantically legal. Whether it is
    // beneficial is a per-backend decision, made by the profitability policy
    // of the xla-inliner pass (see transforms/inliner.cc) rather than by this
    // dialect, which is shared between backends.
    return true;
  }

  // Returns true if the given operation 'op', that is registered to this
  // dialect, can be inlined into the given region, false otherwise.
  // 'wouldBeCloned' is set to true if the given 'op' is set to be cloned
  // during the inlining process, or false if the operation is set to be moved
  // in-place(i.e. no duplicates would be created). 'valueMapping' contains any
  // remapped values from within the 'src' region. This can be used to examine
  // what values may potentially replace the operands to 'op'.
  bool isLegalToInline(mlir::Operation* op, mlir::Region* dest,
                       bool wouldBeCloned,
                       mlir::IRMapping& valueMapping) const final {
    // We allow any op from the xla dialect to be inlined.
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(mlir::Region* dest, mlir::Region* src,
                       bool wouldBeCloned,
                       mlir::IRMapping& valueMapping) const final {
    return true;
  }
};

struct XlaOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(mlir::Attribute attr,
                       mlir::raw_ostream& os) const final {
    if (llvm::isa<IndexingMapAttr>(attr)) {
      os << "indexing_map";
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};

}  // namespace

void XlaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/codegen/emitters/ir/xla_ops.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/codegen/emitters/ir/xla_attrs.cc.inc"
      >();
  addInterfaces<XlaInlinerInterface, XlaOpAsmDialectInterface>();
}

mlir::Operation* XlaDialect::materializeConstant(mlir::OpBuilder& builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  return mlir::arith::ConstantOp::materialize(builder, value, type, loc);
}

}  // namespace xla
