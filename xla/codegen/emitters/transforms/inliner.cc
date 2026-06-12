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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Inliner.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/emitters/type_util.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_XLAINLINERPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

constexpr int64_t kMaxFuncSize = 4000;
constexpr llvm::StringRef kConservativePolicyName = "conservative";
constexpr llvm::StringRef kAggressivePolicyName = "aggressive";

int64_t GetNumOps(mlir::Block& block) { return block.getOperations().size(); }

llvm::StringRef PolicyName(InlinerPolicy policy) {
  switch (policy) {
    case InlinerPolicy::kConservative:
      return kConservativePolicyName;
    case InlinerPolicy::kAggressive:
      return kAggressivePolicyName;
  }
}

std::optional<InlinerPolicy> ParsePolicy(llvm::StringRef name) {
  if (name == kConservativePolicyName) {
    return InlinerPolicy::kConservative;
  }
  if (name == kAggressivePolicyName) {
    return InlinerPolicy::kAggressive;
  }
  return std::nullopt;
}

// Returns true if inlining `callee` at the next call site would clone its
// body rather than move it in place. Mirrors the upstream inliner driver's
// CGUseList::hasOneUseAndDiscardable, which moves the body only for the last
// use of a discardable callee. The driver erases calls as inlining proceeds
// and in-place inlining empties the moved-from body, so a fresh symbol-use
// scan here sees the same use counts the driver's incremental bookkeeping
// does.
bool WouldBeCloned(mlir::func::FuncOp callee) {
  auto symbol = mlir::cast<mlir::SymbolOpInterface>(callee.getOperation());
  if (!symbol.canDiscardOnUseEmpty()) {
    return true;
  }
  auto module = callee->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    return true;
  }
  std::optional<mlir::SymbolTable::UseRange> uses =
      mlir::SymbolTable::getSymbolUses(callee, module.getOperation());
  return !uses || !llvm::hasSingleElement(*uses);
}

// The historical xla emitter inlining heuristic, moved verbatim from
// XlaInlinerInterface::isLegalToInline: inline a callee that would be cloned
// only if it has a single call site in the caller, the result stays under
// kMaxFuncSize, and the caller and callee call a common third function.
bool IsProfitableToInlineConservative(
    const mlir::Inliner::ResolvedCall& resolved_call) {
  mlir::CallOpInterface call = resolved_call.call;
  auto pure_call_op = mlir::dyn_cast<PureCallOp>(call.getOperation());
  // Calls other than xla.pure_call keep the upstream inline pass behavior:
  // always profitable, with legality decided by their own dialect.
  if (!pure_call_op) {
    return true;
  }
  mlir::Region* callable_region = resolved_call.targetNode->getCallableRegion();
  auto func_op =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(callable_region->getParentOp());
  if (!func_op) {
    return true;
  }
  if (func_op->hasAttr(kHasNoCompute)) {
    return true;
  }

  bool would_be_cloned = WouldBeCloned(func_op);

  llvm::SmallDenseSet<llvm::StringRef> callee_calls;
  for (auto callee_call : callable_region->getOps<PureCallOp>()) {
    callee_calls.insert(callee_call.getCallee());
  }

  // If true, then the callee and the caller call the same third function.
  bool contains_call_to_same_function = false;
  // The number of calls to the callee in the caller.
  int num_calls_in_caller = 0;
  if (!would_be_cloned) {
    num_calls_in_caller = 1;
  } else {
    for (auto neighbor_call :
         pure_call_op->getParentRegion()->getOps<PureCallOp>()) {
      contains_call_to_same_function |=
          callee_calls.contains(neighbor_call.getCallee());
      if (neighbor_call.getCallee() == pure_call_op.getCallee()) {
        ++num_calls_in_caller;
      }
    }
  }
  if (num_calls_in_caller > 1) {
    return false;
  }
  // Don't inline functions, if after inlining the size of the function
  // becomes too big.
  int num_ops = num_calls_in_caller * GetNumOps(callable_region->front()) +
                GetNumOps(pure_call_op->getParentRegion()->front());
  if (num_ops > kMaxFuncSize) {
    return false;
  }
  return !would_be_cloned || contains_call_to_same_function;
}

// Inline whenever the caller has a single call site for the callee and the
// result stays under kMaxFuncSize, even if the callee has other callers. A
// call that is not inlined re-evaluates the callee's entire transitive
// computation per use; across call chains whose consecutive levels share no
// callees (e.g. rotation/quaternion chains emitted for kinematic models)
// this recomputation compounds exponentially with chain depth. Inlining
// instead grows code, but the growth is bounded by kMaxFuncSize and
// collapsed by the CSE that runs interleaved with the inliner.
bool IsProfitableToInlineAggressive(
    const mlir::Inliner::ResolvedCall& resolved_call) {
  mlir::CallOpInterface call = resolved_call.call;
  auto pure_call_op = mlir::dyn_cast<PureCallOp>(call.getOperation());
  // Calls other than xla.pure_call keep the upstream inline pass behavior:
  // always profitable, with legality decided by their own dialect.
  if (!pure_call_op) {
    return true;
  }
  mlir::Region* callable_region = resolved_call.targetNode->getCallableRegion();
  auto func_op = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      callable_region->getParentOp());
  if (!func_op) {
    return true;
  }
  if (func_op->hasAttr(kHasNoCompute)) {
    return true;
  }

  // The number of calls to the callee in the caller. Calls to the same
  // callee with distinct arguments are not inlined: there is no
  // recomputation to eliminate and no CSE to collapse the duplicated bodies
  // (identical-argument duplicates are merged by CSE before the inliner sees
  // them).
  int num_calls_in_caller = 0;
  if (!WouldBeCloned(func_op)) {
    num_calls_in_caller = 1;
  } else {
    for (auto neighbor_call :
         pure_call_op->getParentRegion()->getOps<PureCallOp>()) {
      if (neighbor_call.getCallee() == pure_call_op.getCallee()) {
        ++num_calls_in_caller;
      }
    }
  }
  if (num_calls_in_caller > 1) {
    return false;
  }
  int num_ops = num_calls_in_caller * GetNumOps(callable_region->front()) +
                GetNumOps(pure_call_op->getParentRegion()->front());
  return num_ops <= kMaxFuncSize;
}

class XlaInlinerPass : public impl::XlaInlinerPassBase<XlaInlinerPass> {
 public:
  XlaInlinerPass() = default;
  XlaInlinerPass(const XlaInlinerPass& other) = default;
  XlaInlinerPass(InlinerPolicy policy,
                 std::function<void(mlir::OpPassManager&)> default_pipeline)
      : explicit_default_pipeline_(std::move(default_pipeline)) {
    policy_ = std::string(PolicyName(policy));
  }

  void runOnOperation() override {
    std::optional<InlinerPolicy> policy = ParsePolicy(policy_);
    if (!policy) {
      getOperation()->emitError()
          << "unknown xla-inliner policy: '" << policy_ << "'";
      return signalPassFailure();
    }
    mlir::Inliner::ProfitabilityCallbackTy is_profitable_to_inline;
    switch (*policy) {
      case InlinerPolicy::kConservative:
        is_profitable_to_inline = IsProfitableToInlineConservative;
        break;
      case InlinerPolicy::kAggressive:
        is_profitable_to_inline = IsProfitableToInlineAggressive;
        break;
    }

    mlir::InlinerConfig::DefaultPipelineTy default_pipeline;
    if (explicit_default_pipeline_) {
      default_pipeline = explicit_default_pipeline_;
    } else if (!default_pipeline_.empty()) {
      std::string pipeline_str = default_pipeline_;
      default_pipeline = [pipeline_str](mlir::OpPassManager& pm) {
        (void)mlir::parsePassPipeline(pipeline_str, pm);
      };
    }
    mlir::InlinerConfig config(std::move(default_pipeline), max_iterations_);

    mlir::CallGraph& call_graph = getAnalysis<mlir::CallGraph>();
    mlir::Inliner inliner(getOperation(), call_graph, *this,
                          getAnalysisManager(), RunPipelineHelper, config,
                          std::move(is_profitable_to_inline));
    if (mlir::failed(inliner.doInlining())) {
      signalPassFailure();
    }
  }

 private:
  // The runPipeline API is protected within the Pass class, so this helper is
  // required to call it from the inliner driver.
  static mlir::LogicalResult RunPipelineHelper(mlir::Pass& pass,
                                               mlir::OpPassManager& pipeline,
                                               mlir::Operation* op) {
    return mlir::cast<XlaInlinerPass>(pass).runPipeline(pipeline, op);
  }

  // The pipeline run over callables between inlining iterations when the pass
  // is constructed programmatically; takes precedence over the
  // default-pipeline string option.
  std::function<void(mlir::OpPassManager&)> explicit_default_pipeline_;
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateXlaInlinerPass() {
  return std::make_unique<XlaInlinerPass>();
}

std::unique_ptr<mlir::Pass> CreateXlaInlinerPass(
    InlinerPolicy policy,
    std::function<void(mlir::OpPassManager&)> default_pipeline) {
  return std::make_unique<XlaInlinerPass>(policy, std::move(default_pipeline));
}

}  // namespace emitters
}  // namespace xla
