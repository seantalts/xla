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
#include "xla/runtime/buffer_use.h"
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

// Option 1 for callee internals: scratch is pre-assigned per call site in the
// parent and passed as an ordinary caller binding, so the runtime path needs no
// per-call allocation. A callee with an internal temp (private index 2) reads
// param -> temp -> result, with all three indices caller-provided.
TEST(RemappedCallThunkTest, CalleeInternalBufferIsCallerProvidedScratch) {
  auto input = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  auto scratch = LiteralUtil::CreateR1<float>({0.0, 0.0, 0.0, 0.0});
  auto output = LiteralUtil::CreateR1<float>({0.0, 0.0, 0.0, 0.0});

  // Caller buffer space: input@0, scratch@1, output@2.
  BufferAllocations caller_allocations =
      CreateBufferAllocations(input, scratch, output);
  BufferAllocation caller_in = CreateBufferAllocation(0, input);
  BufferAllocation caller_scratch = CreateBufferAllocation(1, scratch);
  BufferAllocation caller_out = CreateBufferAllocation(2, output);

  // Callee private space: param@0, result@1, temp@2. copy 0->2, then 2->1.
  BufferAllocation callee_param = CreateBufferAllocation(0, input);
  BufferAllocation callee_result = CreateBufferAllocation(1, output);
  BufferAllocation callee_temp = CreateBufferAllocation(2, scratch);
  TF_ASSERT_OK_AND_ASSIGN(
      auto copy_in,
      CopyThunk::Create({"copy_in"},
                        CreateBufferAllocationSlice(callee_param), input.shape(),
                        CreateBufferAllocationSlice(callee_temp),
                        scratch.shape()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto copy_out,
      CopyThunk::Create({"copy_out"},
                        CreateBufferAllocationSlice(callee_temp), scratch.shape(),
                        CreateBufferAllocationSlice(callee_result),
                        output.shape()));
  ThunkSequence callee_sequence;
  callee_sequence.push_back(std::move(copy_in));
  callee_sequence.push_back(std::move(copy_out));

  // Bindings indexed by callee allocation index: [0]=param, [1]=result,
  // [2]=temp -- temp drawn from the caller's per-site scratch buffer.
  std::vector<BufferAllocation::Slice> caller_buffers = {
      CreateBufferAllocationSlice(caller_in),
      CreateBufferAllocationSlice(caller_out),
      CreateBufferAllocationSlice(caller_scratch)};

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      RemappedCallThunk::Create({"cu_call"}, std::move(callee_sequence),
                                std::move(caller_buffers)));

  Thunk::ExecuteParams params = {nullptr, &caller_allocations};
  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(output, input);
}

// The thunk scheduler tracks dependencies through buffer_uses(); for a shared
// callee those must be reported in the CALLER's buffer space, not the callee's
// private index space, or the scheduler sees the wrong (private) buffers.
TEST(RemappedCallThunkTest, BufferUsesAreReportedInCallerSpace) {
  auto input = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0});
  auto output = LiteralUtil::CreateR1<float>({0.0, 0.0, 0.0, 0.0});

  // Callee reads private index 0 and writes private index 1.
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

  // Caller provides param at allocation index 2, result at index 3.
  BufferAllocation caller_param = CreateBufferAllocation(2, input);
  BufferAllocation caller_result = CreateBufferAllocation(3, output);
  std::vector<BufferAllocation::Slice> caller_buffers = {
      CreateBufferAllocationSlice(caller_param),
      CreateBufferAllocationSlice(caller_result)};

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      RemappedCallThunk::Create({"cu_call"}, std::move(callee_sequence),
                                std::move(caller_buffers)));

  auto uses = thunk->buffer_uses();
  ASSERT_EQ(uses.size(), 2);

  // Exactly one read of caller index 2 and one write of caller index 3.
  bool read_caller_param = false;
  bool write_caller_result = false;
  for (const BufferUse& use : uses) {
    if (use.access() == BufferUse::MemoryAccess::kRead &&
        use.slice().index() == 2) {
      read_caller_param = true;
    }
    if (use.access() == BufferUse::MemoryAccess::kWrite &&
        use.slice().index() == 3) {
      write_caller_result = true;
    }
  }
  EXPECT_TRUE(read_caller_param);
  EXPECT_TRUE(write_caller_result);
}

}  // namespace
}  // namespace xla::cpu
