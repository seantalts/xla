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

#include <utility>
#include <vector>

#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/copy_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

// A `compilation_unit` callee is emitted ONCE against its own private buffer
// index space (params/result are the callee's own low indices). The same
// emitted sequence is shared by every call site; each site supplies its own
// caller buffers. RemappedCallThunk binds the callee's private indices onto the
// caller's buffers at execute time, so one emitted sequence serves N sites.
//
// To prove the remap actually happens (and isn't accidentally an identity), the
// caller's index space deliberately DIFFERS from the callee's: the caller holds
// the real input/output at indices 2 and 3, while the callee references its own
// indices 0 (param) and 1 (result). A thunk that merely forwarded the caller's
// BufferAllocations unchanged would read/write caller indices 0/1 (junk) and
// leave the real output untouched.
TEST(RemappedCallThunkTest, RemapsCalleePrivateIndicesOntoCallerBuffers) {
  auto junk0 = LiteralUtil::CreateR1<float>({9.0, 9.0, 9.0, 9.0});
  auto junk1 = LiteralUtil::CreateR1<float>({8.0, 8.0, 8.0, 8.0});
  auto input = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  auto output = LiteralUtil::CreateR1<float>({0.0, 0.0, 0.0, 0.0});

  // Caller buffer space: input at index 2, output at index 3.
  BufferAllocations caller_allocations =
      CreateBufferAllocations(junk0, junk1, input, output);
  BufferAllocation caller_in = CreateBufferAllocation(2, input);
  BufferAllocation caller_out = CreateBufferAllocation(3, output);
  BufferAllocation::Slice caller_in_slice =
      CreateBufferAllocationSlice(caller_in);
  BufferAllocation::Slice caller_out_slice =
      CreateBufferAllocationSlice(caller_out);

  // Callee (shared, emitted once) in its OWN index space: copy index 0 -> 1.
  BufferAllocation callee_param = CreateBufferAllocation(0, input);
  BufferAllocation callee_result = CreateBufferAllocation(1, output);
  TF_ASSERT_OK_AND_ASSIGN(
      auto copy, CopyThunk::Create({"copy"},
                                   CreateBufferAllocationSlice(callee_param),
                                   input.shape(),
                                   CreateBufferAllocationSlice(callee_result),
                                   output.shape()));
  ThunkSequence callee_sequence;
  callee_sequence.push_back(std::move(copy));

  // Bind callee index i -> caller buffer. Order matches the callee's index
  // space: [0] = param, [1] = result.
  std::vector<BufferAllocation::Slice> caller_buffers = {caller_in_slice,
                                                         caller_out_slice};

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, RemappedCallThunk::Create({"cu_call"},
                                            std::move(callee_sequence),
                                            std::move(caller_buffers)));

  Thunk::ExecuteParams params = {nullptr, &caller_allocations};
  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  // The shared callee wrote through the remapped caller result buffer.
  EXPECT_EQ(output, input);
}

}  // namespace
}  // namespace xla::cpu
