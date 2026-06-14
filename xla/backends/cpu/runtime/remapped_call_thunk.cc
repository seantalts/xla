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

#include "xla/backends/cpu/runtime/remapped_call_thunk.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/constant_allocation.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<RemappedCallThunk>> RemappedCallThunk::Create(
    Info info, std::shared_ptr<ThunkExecutor> called_executor,
    std::vector<BufferAllocation::Slice> caller_buffers,
    std::shared_ptr<const std::vector<ConstantAllocation>> constants) {
  if (called_executor == nullptr) {
    return absl::InvalidArgumentError(
        "RemappedCallThunk requires a non-null called_executor");
  }

  // A null binding entry is only legal if a thunk-owned constant covers that
  // callee index; otherwise the index is unresolvable at execute time.
  absl::flat_hash_set<BufferAllocation::Index> constant_indices;
  if (constants != nullptr) {
    for (const ConstantAllocation& c : *constants) {
      constant_indices.insert(c.index);
    }
  }
  for (int64_t i = 0; i < static_cast<int64_t>(caller_buffers.size()); ++i) {
    if (caller_buffers[i].allocation() == nullptr &&
        !constant_indices.contains(i)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Callee allocation %d has no caller binding and no constant", i));
    }
  }

  return absl::WrapUnique(new RemappedCallThunk(
      std::move(info), std::move(called_executor), std::move(caller_buffers),
      std::move(constants)));
}

RemappedCallThunk::RemappedCallThunk(
    Info info, std::shared_ptr<ThunkExecutor> called_executor,
    std::vector<BufferAllocation::Slice> caller_buffers,
    std::shared_ptr<const std::vector<ConstantAllocation>> constants)
    : Thunk(Kind::kCall, std::move(info)),
      called_executor_(std::move(called_executor)),
      caller_buffers_(std::move(caller_buffers)),
      constants_(std::move(constants)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> RemappedCallThunk::Execute(
    const ExecuteParams& params) {
  const BufferAllocations* caller = params.buffer_allocations;

  // Resolve the callee's private allocation index space onto the caller's
  // buffers: callee index `i` reads/writes the caller buffer `caller_buffers_[i]`.
  // Callee constant indices have no caller buffer; they are filled from the
  // thunk-owned ConstantAllocations and skipped in the caller-binding pass.
  BufferAllocations::Buffers callee_buffers(caller_buffers_.size());
  if (constants_ != nullptr) {
    for (const ConstantAllocation& c : *constants_) {
      callee_buffers[c.index] = c.AsDeviceAddress();
    }
  }
  for (int64_t i = 0; i < static_cast<int64_t>(caller_buffers_.size()); ++i) {
    if (caller_buffers_[i].allocation() == nullptr) {
      continue;  // Constant: already filled above.
    }
    TF_ASSIGN_OR_RETURN(callee_buffers[i],
                        caller->GetDeviceAddress(caller_buffers_[i]));
  }

  // The callee-space allocations must outlive the (possibly async) nested
  // execution, so hold them in a shared_ptr kept alive by the returned event.
  auto callee_allocations =
      std::make_shared<BufferAllocations>(std::move(callee_buffers));

  ExecuteParams callee_params = params;
  callee_params.buffer_allocations = callee_allocations.get();

  auto event = called_executor_->Execute(callee_params);
  event.AndThen([callee_allocations = std::move(callee_allocations)] {});
  return event;
}

RemappedCallThunk::BufferUses RemappedCallThunk::buffer_uses() const {
  // The callee's thunks report uses in the callee's private index space. Map
  // each one through the binding to the caller buffer it actually touches so the
  // parent scheduler tracks the real dependencies. Offsets compose: a callee use
  // at offset o within callee allocation i lands at caller_buffers_[i].offset()
  // + o in the caller's buffer.
  BufferUses uses;
  for (const BufferUse& use : called_executor_->buffer_uses()) {
    const BufferAllocation::Slice& callee_slice = use.slice();
    const BufferAllocation::Slice& caller = caller_buffers_[callee_slice.index()];
    if (caller.allocation() == nullptr) {
      // Callee constant: no caller buffer to map, so report no caller-space use.
      continue;
    }
    BufferAllocation::Slice mapped(caller.allocation(),
                                   caller.offset() + callee_slice.offset(),
                                   callee_slice.size(), caller.element_type());
    uses.push_back(BufferUse(mapped, use.access(), use.content_validity(),
                             use.shape()));
  }
  return uses;
}

RemappedCallThunk::ResourceUses RemappedCallThunk::resource_uses() const {
  return called_executor_->resource_uses();
}

std::vector<std::pair<std::string, const ThunkSequence*>>
RemappedCallThunk::nested_thunks() const {
  return {{absl::StrCat(info().op_name, "-called_sequence"),
           &called_executor_->thunk_sequence()}};
}

}  // namespace xla::cpu
