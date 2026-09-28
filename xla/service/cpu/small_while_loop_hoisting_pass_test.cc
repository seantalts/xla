/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/small_while_loop_hoisting_pass.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/cpu/backend_config.pb.h"

namespace xla {
namespace {

class SmallWhileLoopHoistingPassTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunSmallWhileLoopHoistingPass(HloModule* module) {
    return cpu::SmallWhileLoopHoistingPass(1024).Run(module);
  }
};

TEST_F(SmallWhileLoopHoistingPassTest, SmallWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule simple_while_loop

    while_body {
      counter = s32[] parameter(0)
      increment = s32[] constant(1)
      ROOT incremented_counter = s32[] add(counter, increment)
    }

    while_condition {
      counter = s32[] parameter(0)
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      initial_counter = s32[] constant(0)
      ROOT while_loop = s32[] while(initial_counter), condition=while_condition, body=while_body
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call_instr = FindInstruction(m.get(), HloOpcode::kCall);
  ASSERT_NE(call_instr, nullptr);
  std::optional<std::string> maybe_small_call =
      call_instr->get_frontend_attribute("xla_cpu_small_call");
  ASSERT_NE(maybe_small_call, std::nullopt);
  EXPECT_EQ(*maybe_small_call, "true");

  EXPECT_EQ(call_instr->to_apply()->root_instruction()->opcode(),
            HloOpcode::kWhile);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoBigWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule simple_while_loop

    reduce_fn {
      x = s32[] parameter(0)
      y = s32[] parameter(1)
      ROOT add = s32[] add(x, y)
    }

    while_body {
      counter = s32[] parameter(0)
      dummy_constant = s32[1000000] constant({...})
      // The big constant must be in the call graph to be considered in the cost
      // analysis, hence the reduce.
      element_reduce = s32[] reduce(dummy_constant, counter), dimensions={0}, to_apply=reduce_fn
      ROOT incremented_counter = s32[] add(counter, element_reduce)
    }

    while_condition {
      counter = s32[] parameter(0)
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      initial_counter = s32[] constant(0)
      ROOT while_loop = s32[] while(initial_counter), condition=while_condition, body=while_body
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoInOutFeedWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule in_out_feed_while_loop, entry_computation_layout={(pred[])->(pred[])}

    body_fn (T.4: (pred[])) -> (pred[]) {
      T.4 = (pred[]) parameter(0)
      after-all.5 = token[] after-all()
      infeed.6 = ((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed(token[] after-all.5)
      get-tuple-element.7 = token[] get-tuple-element(((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed.6), index=1
      get-tuple-element.8 = (f32[1,3]{1,0}, pred[], u32[]) get-tuple-element(((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed.6), index=0
      get-tuple-element.11 = f32[1,3]{1,0} get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=0
      constant.12 = f32[] constant(1)
      broadcast.13 = f32[1,3]{1,0} broadcast(f32[] constant.12), dimensions={}
      multiply.14 = f32[1,3]{1,0} multiply(f32[1,3]{1,0} get-tuple-element.11, f32[1,3]{1,0} broadcast.13)
      concatenate.15 = f32[1,6]{1,0} concatenate(f32[1,3]{1,0} multiply.14, f32[1,3]{1,0} multiply.14), dimensions={1}
      get-tuple-element.10 = u32[] get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=2
      tuple.16 = (f32[1,6]{1,0}, u32[]) tuple(f32[1,6]{1,0} concatenate.15, u32[] get-tuple-element.10)
      after-all.17 = token[] after-all()
      outfeed.18 = token[] outfeed((f32[1,6]{1,0}, u32[]) tuple.16, token[] after-all.17), outfeed_shape=(f32[1,6]{1,0}, u32[])
      tuple.19 = () tuple()
      get-tuple-element.9 = pred[] get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=1
      ROOT tuple.20 = (pred[]) tuple(pred[] get-tuple-element.9)
    }

    condition_fn (T.22: (pred[])) -> pred[] {
      T.22 = (pred[]) parameter(0)
      ROOT get-tuple-element.23 = pred[] get-tuple-element((pred[]) T.22), index=0
    }

    ENTRY main (prev0.1: pred[]) -> (pred[]) {
      prev0.1 = pred[] parameter(0)
      tuple.2 = (pred[]) tuple(pred[] prev0.1)
      ROOT tuple.26 = (pred[]) while((pred[]) tuple.2), condition=condition_fn, body=body_fn
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoFftWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule fft_module

    %body_comp (arg_tuple.3: (s32[], c64[30])) -> (s32[], c64[30]) {
      %arg_tuple.3 = (s32[], c64[30]{0}) parameter(0)
      %get-tuple-element.4 = s32[] get-tuple-element(%arg_tuple.3), index=0
      %constant.6 = s32[] constant(1)
      %add.14 = s32[] add(%get-tuple-element.4, %constant.6)
      %get-tuple-element.5 = c64[30]{0} get-tuple-element(%arg_tuple.3), index=1
      %fft.10 = c64[30]{0} fft(%get-tuple-element.5), fft_type=FFT, fft_length={30}
      ROOT %tuple.15 = (s32[], c64[30]{0}) tuple(%add.14, %get-tuple-element.5)
    }

    %condition_comp (arg_tuple.17: (s32[], c64[30])) -> pred[] {
      %arg_tuple.17 = (s32[], c64[30]{0}) parameter(0)
      %get-tuple-element.18 = s32[] get-tuple-element(%arg_tuple.17), index=0
      %constant.20 = s32[] constant(10)
      ROOT %lt.21 = pred[] compare(%get-tuple-element.18, %constant.20), direction=LT
    }

    ENTRY %main.27 (args_0_.1: c64[30]) -> c64[30] {
      %constant.2 = s32[] constant(0)
      %args_0_.1 = c64[30]{0} parameter(0)
      %while.23 = (s32[], c64[30]{0}) tuple(%constant.2, %args_0_.1)
      %while.24 = (s32[], c64[30]{0}) while(%while.23), condition=%condition_comp, body=%body_comp
      %while.25 = s32[] get-tuple-element(%while.24), index=0
      ROOT %while.26 = c64[30]{0} get-tuple-element(%while.24), index=1
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, NoYnnWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule ynn_module

    %ynn_comp (lhs: f32[8,3], rhs: f32[3,3]) -> f32[8,3] {
      %lhs = f32[8,3] parameter(0)
      %rhs = f32[3,3] parameter(1)
      ROOT %dot = f32[8,3] dot(%lhs, %rhs), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}
    }

    %closed_call (x: f32[8,3]) -> (f32[8,3], f32[3], f32[3,3]) {
      %x = f32[8,3] parameter(0)
      %one = f32[] constant(1)
      %one_2d = f32[3,3] broadcast(%one), dimensions={}
      %ynn_fusion = f32[8,3] fusion(%x, %one_2d), kind=kCustom, calls=%ynn_comp,
        backend_config={
            "outer_dimension_partitions":[],
            "fusion_config":{"kind":"__ynn_fusion"}
          }
      %zero = f32[] constant(0)
      %zero_1d = f32[3]{0} broadcast(%zero), dimensions={}
      ROOT %tuple = (f32[8,3], f32[3], f32[3,3]) tuple(%ynn_fusion,
                                                       %zero_1d, %one_2d)
    }

    %body_comp (state: (s32[], f32[8,3], f32[8,3], f32[8,3,3])) ->
                       (s32[], f32[8,3], f32[8,3], f32[8,3,3]) {
      %state = (s32[], f32[8,3], f32[8,3], f32[8,3,3]) parameter(0)
      %idx = s32[] get-tuple-element(%state), index=0
      %one = s32[] constant(1)
      %new_idx = s32[] add(%idx, %one)
      %x = f32[8,3] get-tuple-element(%state), index=1
      %call = (f32[8,3], f32[3], f32[3,3]) call(%x), to_apply=%closed_call
      %in2 = f32[8,3] get-tuple-element(%state), index=2
      %in3 = f32[8,3,3] get-tuple-element(%state), index=3
      ROOT tuple = (s32[], f32[8,3], f32[8,3], f32[8,3,3])
                      tuple(%new_idx, %x, %in2, %in3)
    }

    %cond_comp (state: (s32[], f32[8,3], f32[8,3], f32[8,3,3])) -> pred[] {
      %state = (s32[], f32[8,3], f32[8,3], f32[8,3,3]) parameter(0)
      %idx = s32[] get-tuple-element(%state), index=0
      %eight = s32[] constant(8)
      ROOT %lt.1 = pred[] compare(%idx, %eight), direction=LT
    }

    ENTRY %main (x: f32[8,3]) -> (f32[8,3], f32[8,3], f32[8,3,3]) {
      %zero_int = s32[] constant(0)
      %x = f32[8,3] parameter(0)
      %zero_float = f32[] constant(0)
      %zero_2d = f32[8,3] broadcast(%zero_float), dimensions={}
      %zero_3d = f32[8,3,3] broadcast(%zero_float), dimensions={}
      %init_state = (s32[], f32[8,3], f32[8,3], f32[8,3,3])
                      tuple(%zero_int, %x, %zero_2d, %zero_3d)
      %while = (s32[], f32[8,3], f32[8,3], f32[8,3,3])
                  while(%init_state), condition=%cond_comp,
                  body=%body_comp
      %res_2d = f32[8,3] get-tuple-element(%while), index=2
      %res_3d = f32[8,3,3] get-tuple-element(%while), index=3
      ROOT %tuple = (f32[8,3], f32[8,3], f32[8,3,3]) tuple(%x, %res_2d, %res_3d)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallWhileLoopHoistingPassTest, UnimplementedOpcodesAreUnavailable) {
  constexpr HloOpcode kOpcodes[] = {
      HloOpcode::kBatchNormGrad,
      HloOpcode::kBatchNormTraining,
      HloOpcode::kCustomCall,
      HloOpcode::kFft,
      HloOpcode::kGetDimensionSize,
      HloOpcode::kInfeed,
      HloOpcode::kOutfeed,
      HloOpcode::kPartitionId,
      HloOpcode::kRecv,
      HloOpcode::kRecvDone,
      HloOpcode::kReplicaId,
      HloOpcode::kRng,
      HloOpcode::kRngBitGenerator,
      HloOpcode::kScatter,
      HloOpcode::kSend,
      HloOpcode::kSendDone,
      HloOpcode::kSetDimensionSize,
      HloOpcode::kSort,
      HloOpcode::kStochasticConvert,
      HloOpcode::kTopK,
  };
  for (HloOpcode opcode : kOpcodes) {
    EXPECT_TRUE(cpu::IsUnavailableOpcodeInHoistedRegion(opcode))
        << HloOpcodeString(opcode);
  }
  EXPECT_FALSE(cpu::IsUnavailableOpcodeInHoistedRegion(HloOpcode::kAdd));
  EXPECT_FALSE(cpu::IsUnavailableOpcodeInHoistedRegion(
      HloOpcode::kRngGetAndUpdateState));
}

TEST_F(SmallWhileLoopHoistingPassTest, UnavailableInstructionsInWhileBody) {
  struct Case {
    absl::string_view name;
    absl::string_view hlo;
  };
  const std::vector<Case> kCases = {
      {"ar", "ar = f32[8] all-reduce(p), to_apply=add"},
      {"rs",
       "rs = f32[4] reduce-scatter(p), dimensions={0}, to_apply=add, "
       "replica_groups={{0,1}}"},
      {"ag",
       "ag = f32[16] all-gather(p), dimensions={0}, replica_groups={{0,1}}"},
      {"a2a",
       "a2a = f32[8] all-to-all(p), dimensions={0}, replica_groups={{0,1}}"},
      {"cp",
       "cp = f32[8] collective-permute(p), source_target_pairs={{0,1},{1,0}}"},
      {"pid", "pid = u32[] partition-id()"},
      {"rid", "rid = u32[] replica-id()"},
      {"cc", "cc = f32[8] custom-call(p), custom_call_target=\"foo\""},
      {"srt", "srt = f32[8] sort(p), dimensions={0}, to_apply=cmp"},
      {"infd", "infd = (f32[8], token[]) infeed(tok)"},
      {"outf", "outf = token[] outfeed(p, tok), outfeed_shape=f32[8]"},
      {"snd",
       "snd = (f32[8], u32[], token[]) send(p, tok), channel_id=1\n"
       "sd = token[] send-done(snd), channel_id=1"},
      {"rcv",
       "rcv = (f32[8], u32[], token[]) recv(tok), channel_id=2\n"
       "rd = (f32[8], token[]) recv-done(rcv), channel_id=2"},
      {"cf",
       "cf = f32[8] fusion(p), kind=kCustom, calls=custom_fusion_computation"},
  };
  for (const Case& c : kCases) {
    const std::string hlo = absl::StrCat(R"(
    HloModule m, replica_count=2

    add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT s = f32[] add(x, y)
    }

    cmp {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT lt = pred[] compare(x, y), direction=LT
    }

    custom_fusion_computation {
      x = f32[8] parameter(0)
      ROOT n = f32[8] negate(x)
    }

    body {
      t = (s32[], f32[8]) parameter(0)
      i = s32[] get-tuple-element(t), index=0
      p = f32[8] get-tuple-element(t), index=1
      one = s32[] constant(1)
      tok = token[] after-all()
      )",
                                         c.hlo, R"(
      inc = s32[] add(i, one)
      ROOT r = (s32[], f32[8]) tuple(inc, p)
    }

    cond {
      t = (s32[], f32[8]) parameter(0)
      i = s32[] get-tuple-element(t), index=0
      limit = s32[] constant(10)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      arg = f32[8] parameter(0)
      init = (s32[], f32[8]) tuple(zero, arg)
      ROOT w = (s32[], f32[8]) while(init), condition=cond, body=body
    }
    )");
    SCOPED_TRACE(c.hlo);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo));
    const HloInstruction* instr = FindInstruction(m.get(), c.name);
    ASSERT_NE(instr, nullptr);
    EXPECT_TRUE(cpu::IsUnavailableInHoistedRegion(instr));
    ASSERT_OK_AND_ASSIGN(bool changed, RunSmallWhileLoopHoistingPass(m.get()));
    EXPECT_FALSE(changed);
  }
}

}  // namespace
}  // namespace xla
