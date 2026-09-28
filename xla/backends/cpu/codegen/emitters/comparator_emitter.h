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

#ifndef XLA_BACKENDS_CPU_CODEGEN_EMITTERS_COMPARATOR_EMITTER_H_
#define XLA_BACKENDS_CPU_CODEGEN_EMITTERS_COMPARATOR_EMITTER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_computation.h"

namespace xla::cpu {

// Emits an MLIR module with a single public function
//
//   func.func @<function_name>(%data: !llvm.ptr) -> i1
//
// implementing the `sort` comparator computation. `data` points to an array
// of `num_parameters` pointers; entry `k` points to the scalar value of the
// comparator's k-th parameter. The module is ready for the scalar
// FusionCompiler pipeline, which produces `bool(const void** data)` with a
// zero-extended result.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitComparatorModule(
    mlir::MLIRContext& context, const HloComputation& comparator,
    absl::string_view function_name);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_COMPARATOR_EMITTER_H_
