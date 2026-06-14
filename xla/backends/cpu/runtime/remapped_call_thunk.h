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

#ifndef XLA_BACKENDS_CPU_RUNTIME_REMAPPED_CALL_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_REMAPPED_CALL_THUNK_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/constant_allocation.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Calls a SHARED callee thunk sequence whose thunks reference buffers in the
// callee's own private allocation index space, remapping those indices onto the
// caller's buffers at execute time.
//
// This is the runtime primitive behind XLA:CPU `compilation_unit` sharing: a
// marked computation (e.g. a loop body unrolled into N call sites) is emitted
// ONCE against a private buffer assignment, and each call site invokes it with
// its own caller buffers instead of forcing N cloned copies.
class RemappedCallThunk final : public Thunk {
 public:
  // `caller_buffers[i]` is the caller buffer that provides callee allocation
  // index `i` (parameters and result occupy the callee's low indices).
  //
  // The callee `ThunkExecutor` is shared (re-entrant: each Execute builds its own
  // per-run heap state), so N call sites of one compilation unit can hold the
  // same `called_executor` and bind it onto their own `caller_buffers`.
  //
  // A null (default-constructed) `caller_buffers[i]` slice means callee index `i`
  // is a callee constant: there is no caller buffer for it (the runtime buffer
  // table is sized to the parent assignment only). Such an index must be covered
  // by an entry in `constants` with that index; the thunk owns the constants and
  // resolves those indices itself. `constants` may be nullptr.
  static absl::StatusOr<std::unique_ptr<RemappedCallThunk>> Create(
      Info info, std::shared_ptr<ThunkExecutor> called_executor,
      std::vector<BufferAllocation::Slice> caller_buffers,
      std::shared_ptr<const std::vector<ConstantAllocation>> constants);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;
  ResourceUses resource_uses() const final;

  std::vector<std::pair<std::string, const ThunkSequence*>> nested_thunks()
      const final;

 private:
  RemappedCallThunk(
      Info info, std::shared_ptr<ThunkExecutor> called_executor,
      std::vector<BufferAllocation::Slice> caller_buffers,
      std::shared_ptr<const std::vector<ConstantAllocation>> constants);

  std::shared_ptr<ThunkExecutor> called_executor_;
  std::vector<BufferAllocation::Slice> caller_buffers_;
  std::shared_ptr<const std::vector<ConstantAllocation>> constants_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_REMAPPED_CALL_THUNK_H_
