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

// Diagnostic benchmark: load an HLO text file, compile once for CPU, execute
// N times and report per-iteration wall time. With --profile, additionally
// runs ~10 iterations under a tsl::profiler::ProfilerSession and prints the
// top event names (e.g. per-thunk TraceMes) by total duration.

#include <algorithm>
#include <cstdint>
#include <random>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/run_hlo_module.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/env.h"
#include "tsl/platform/init_main.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {

const char* const kUsage = R"(
Loads an HLO text file, runs HLO passes and compiles it ONCE for the CPU
backend, then executes it repeatedly and reports per-iteration wall time.

Usage:
  bench_hlo --hlo_file=path/to/module.hlo [--iters=100] [--warmup=5] [--profile]

The HLO file may also be passed as a positional argument. XLA_FLAGS is
honored.
)";

struct EventStat {
  int64_t count = 0;
  int64_t total_ps = 0;
};

void PrintTopEvents(const tensorflow::profiler::XSpace& space, int top_n) {
  absl::flat_hash_map<std::string, EventStat> stats;
  for (const auto& plane : space.planes()) {
    for (const auto& line : plane.lines()) {
      for (const auto& event : line.events()) {
        auto it = plane.event_metadata().find(event.metadata_id());
        if (it == plane.event_metadata().end()) continue;
        EventStat& s = stats[it->second.name()];
        ++s.count;
        s.total_ps += event.duration_ps();
      }
    }
  }
  std::vector<std::pair<std::string, EventStat>> sorted(stats.begin(),
                                                        stats.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.second.total_ps > b.second.total_ps;
  });
  std::cout << "PROFILE: top " << top_n << " events by total duration\n";
  std::cout << absl::StrFormat("%-12s %8s  %s\n", "total_ms", "count", "name");
  int printed = 0;
  for (const auto& [name, s] : sorted) {
    if (printed++ >= top_n) break;
    std::cout << absl::StrFormat("%-12.3f %8d  %s\n", s.total_ps / 1e9,
                                 s.count, name);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string hlo_file;
  int32_t iters = 100;
  int32_t warmup = 5;
  bool profile = false;
  bool run_hlo_passes = true;
  bool print_result = false;
  int64_t seed = -1;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo_file", &hlo_file, "Path to HLO text file."),
      tsl::Flag("iters", &iters, "Number of timed iterations."),
      tsl::Flag("warmup", &warmup, "Number of untimed warmup iterations."),
      tsl::Flag("profile", &profile,
                "Run ~10 extra iterations under a profiler session and print "
                "per-event (e.g. per-thunk) time attribution."),
      tsl::Flag("run_hlo_passes", &run_hlo_passes,
                "Run the HLO pass pipeline before compiling. Set to false for "
                "already-optimized HLO."),
      tsl::Flag("print_result", &print_result,
                "Print the result literal once (for correctness checks)."),
      tsl::Flag("seed", &seed,
                "If >= 0, seed fake-argument generation deterministically so "
                "results are reproducible across processes (for bitwise "
                "flag-on/off parity checks). Default -1 = nondeterministic."),
  };
  xla::AppendDebugOptionsFlags(&flag_list);
  xla::ParseDebugOptionFlagsFromEnv(true);

  const std::string usage =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(usage.c_str(), &argc, &argv);
  if (!parse_ok) {
    std::cerr << usage;
    return 1;
  }
  if (hlo_file.empty() && argc > 1) {
    hlo_file = argv[1];
  }
  QCHECK(!hlo_file.empty()) << "Must pass --hlo_file or a positional HLO file.";

  // Parse the module with debug options from flags/XLA_FLAGS.
  std::string hlo_text;
  TF_QCHECK_OK(
      tsl::ReadFileToString(tsl::Env::Default(), hlo_file, &hlo_text));
  xla::HloModuleConfig config;
  config.set_debug_options(xla::GetDebugOptionsFromFlags());
  absl::StatusOr<std::unique_ptr<xla::HloModule>> module =
      xla::ParseAndReturnUnverifiedModule(hlo_text, config);
  TF_QCHECK_OK(module.status()) << "Failed to parse " << hlo_file;

  // Fake arguments, generated once from the unoptimized entry signature. With
  // --seed >= 0 the generator is seeded so separate processes (flag on vs off)
  // produce identical inputs, enabling a bitwise output-parity check.
  absl::StatusOr<std::vector<xla::Literal>> args;
  if (seed >= 0) {
    std::minstd_rand0 engine(static_cast<std::minstd_rand0::result_type>(seed));
    args = xla::MakeFakeArguments(module->get(), &engine);
  } else {
    args = xla::MakeFakeArguments(module->get());
  }
  TF_QCHECK_OK(args.status());

  // CPU runner; compile exactly once (with HLO passes).
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client =
      xla::GetPjRtClientForPlatform("cpu");
  TF_QCHECK_OK(client.status());
  xla::HloRunnerPjRt runner(*std::move(client));

  const absl::Time compile_start = absl::Now();
  absl::StatusOr<std::unique_ptr<xla::OpaqueExecutable>> executable =
      runner.CreateExecutable(*std::move(module), run_hlo_passes);
  TF_QCHECK_OK(executable.status());
  std::cout << "Compiled " << hlo_file << " in "
            << absl::ToDoubleMilliseconds(absl::Now() - compile_start)
            << " ms\n";

  auto run_once = [&]() {
    absl::StatusOr<xla::Literal> result =
        runner.ExecuteWithExecutable(executable->get(), *args);
    TF_QCHECK_OK(result.status());
  };

  if (print_result) {
    absl::StatusOr<xla::Literal> result =
        runner.ExecuteWithExecutable(executable->get(), *args);
    TF_QCHECK_OK(result.status());
    std::cout << "RESULT: " << result->ToString() << "\n";
  }

  for (int i = 0; i < warmup; ++i) run_once();

  std::vector<double> times_ms;
  times_ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    const absl::Time start = absl::Now();
    run_once();
    times_ms.push_back(absl::ToDoubleMilliseconds(absl::Now() - start));
  }

  double mean = 0.0;
  for (double t : times_ms) mean += t;
  mean /= times_ms.empty() ? 1 : times_ms.size();
  std::vector<double> sorted_ms = times_ms;
  std::sort(sorted_ms.begin(), sorted_ms.end());
  const double median =
      sorted_ms.empty()
          ? 0.0
          : (sorted_ms.size() % 2 == 1
                 ? sorted_ms[sorted_ms.size() / 2]
                 : (sorted_ms[sorted_ms.size() / 2 - 1] +
                    sorted_ms[sorted_ms.size() / 2]) /
                       2.0);
  std::cout << absl::StrFormat("BENCH: n=%d mean_ms=%.3f median_ms=%.3f\n",
                               iters, mean, median);

  if (profile) {
    constexpr int kProfileIters = 10;
    tensorflow::ProfileOptions options =
        tsl::ProfilerSession::DefaultOptions();
    std::unique_ptr<tsl::ProfilerSession> session =
        tsl::ProfilerSession::Create(options);
    TF_QCHECK_OK(session->Status());
    for (int i = 0; i < kProfileIters; ++i) run_once();
    tensorflow::profiler::XSpace space;
    TF_QCHECK_OK(session->CollectData(&space));
    std::cout << "PROFILE: aggregated over " << kProfileIters
              << " iterations\n";
    PrintTopEvents(space, /*top_n=*/30);
  }

  return 0;
}
