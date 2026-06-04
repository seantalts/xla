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

#ifndef XLA_SERVICE_MULTI_MODULE_DRIVER_H_
#define XLA_SERVICE_MULTI_MODULE_DRIVER_H_

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"

namespace tsl {
class Executor;
}  // namespace tsl

namespace xla {

// Orchestrates the splitting of an HLO module into multiple modules and their
// compilation. This is used when some computations are marked as
// non-inlineable.
class MultiModuleDriver {
 public:
  using CompileFn = std::function<absl::StatusOr<std::unique_ptr<HloModule>>(
      std::unique_ptr<HloModule>, const Compiler::CompileOptions&)>;

  explicit MultiModuleDriver(CompileFn compile_fn,
                             tsl::Executor* executor = nullptr)
      : compile_fn_(std::move(compile_fn)), executor_(executor) {}

  // Returns true if the module contains any non-inlineable computations and
  // should be processed by the multi-module driver.
  static bool ShouldProcess(const HloModule& module);

  // Splits the module, runs HLO passes on submodules concurrently, stitches
  // them back, and returns the unified module.
  // The `compile_fn` callback must be thread-safe when concurrent compilation
  // is enabled (by passing a thread pool in `options` or during driver
  // creation).
  absl::StatusOr<std::unique_ptr<HloModule>> Compile(
      std::unique_ptr<HloModule> module,
      const std::vector<se::StreamExecutor*>& stream_execs,
      const Compiler::CompileOptions& options) const;

  // An optimized parent module plus the optimized submodules extracted from it.
  struct SplitCompileResult {
    // Parent module. When `submodules` is non-empty it is left UN-stitched: the
    // call sites remain `kMultiModuleCustomCallTarget` custom-calls referencing
    // the submodules by name. (Empty `submodules` means nothing was split.)
    std::unique_ptr<HloModule> module;
    // Optimized, deduplicated submodules (one per unique compilation unit).
    std::vector<std::unique_ptr<HloModule>> submodules;
  };

  // Like Compile, but does NOT stitch the optimized submodules back into the
  // parent: returns the parent with custom-calls intact plus the submodules, so
  // the backend can emit each submodule once and lower the custom-calls to
  // shared calls. This is the entry point for shared compilation-unit emission.
  absl::StatusOr<SplitCompileResult> CompileKeepingSubmodules(
      std::unique_ptr<HloModule> module,
      const std::vector<se::StreamExecutor*>& stream_execs,
      const Compiler::CompileOptions& options) const;

  // Returns the number of times Compile() has been called. Used for testing
  // and debugging purposes.
  static int GetCompileCount();

  // Resets the compile counter to 0. Used for testing and debugging purposes.
  static void ResetCompileCount();

 private:
  // Splits `module` and compiles the parent + each submodule via `compile_fn_`
  // (in parallel when an executor is available). Returns the optimized parent
  // and optimized submodules, UN-stitched. Empty `submodules` means the module
  // had no compilation units to split.
  absl::StatusOr<SplitCompileResult> SplitAndCompile(
      std::unique_ptr<HloModule> module,
      const Compiler::CompileOptions& options) const;

  CompileFn compile_fn_;
  tsl::Executor* executor_;
  static std::atomic<int> compile_count_;
};

}  // namespace xla

#endif  // XLA_SERVICE_MULTI_MODULE_DRIVER_H_
