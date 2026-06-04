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

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
    Info info, ThunkSequence called_sequence,
    std::vector<BufferAllocation::Slice> caller_buffers) {
  TF_ASSIGN_OR_RETURN(auto called_executor,
                      ThunkExecutor::Create(std::move(called_sequence)));
  return absl::WrapUnique(new RemappedCallThunk(
      std::move(info), std::move(called_executor), std::move(caller_buffers)));
}

RemappedCallThunk::RemappedCallThunk(
    Info info, ThunkExecutor called_executor,
    std::vector<BufferAllocation::Slice> caller_buffers)
    : Thunk(Kind::kCall, std::move(info)),
      called_executor_(std::move(called_executor)),
      caller_buffers_(std::move(caller_buffers)) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> RemappedCallThunk::Execute(
    const ExecuteParams& params) {
  const BufferAllocations* caller = params.buffer_allocations;

  // Resolve the callee's private allocation index space onto the caller's
  // buffers: callee index `i` reads/writes the caller buffer `caller_buffers_[i]`.
  BufferAllocations::Buffers callee_buffers;
  callee_buffers.reserve(caller_buffers_.size());
  for (const BufferAllocation::Slice& slice : caller_buffers_) {
    TF_ASSIGN_OR_RETURN(se::DeviceAddressBase address,
                        caller->GetDeviceAddress(slice));
    callee_buffers.push_back(address);
  }

  // The callee-space allocations must outlive the (possibly async) nested
  // execution, so hold them in a shared_ptr kept alive by the returned event.
  auto callee_allocations =
      std::make_shared<BufferAllocations>(std::move(callee_buffers));

  ExecuteParams callee_params = params;
  callee_params.buffer_allocations = callee_allocations.get();

  auto event = called_executor_.Execute(callee_params);
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
  for (const BufferUse& use : called_executor_.buffer_uses()) {
    const BufferAllocation::Slice& callee_slice = use.slice();
    const BufferAllocation::Slice& caller = caller_buffers_[callee_slice.index()];
    BufferAllocation::Slice mapped(caller.allocation(),
                                   caller.offset() + callee_slice.offset(),
                                   callee_slice.size(), caller.element_type());
    uses.push_back(BufferUse(mapped, use.access(), use.content_validity(),
                             use.shape()));
  }
  return uses;
}

RemappedCallThunk::ResourceUses RemappedCallThunk::resource_uses() const {
  return called_executor_.resource_uses();
}

std::vector<std::pair<std::string, const ThunkSequence*>>
RemappedCallThunk::nested_thunks() const {
  return {{absl::StrCat(info().op_name, "-called_sequence"),
           &called_executor_.thunk_sequence()}};
}

}  // namespace xla::cpu
