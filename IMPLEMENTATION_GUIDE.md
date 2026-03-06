# Two-Phase Fusion for XLA CPU: Complete Implementation Guide

## Problem Statement

The XLA CPU compiler has a fusion pipeline that runs on the entry computation
after inlining sub-computations. For large HLO modules with many outlined
sub-computations (e.g., JAX's MuJoCo MJX which has ~200 sub-computations),
inlining first creates a massive flat graph that the fusion pass must then
process. This leads to quadratic-or-worse compile times because the fusion
pass repeatedly scans the huge flattened graph.

The fix: **fuse within each sub-computation first** (compacting them), **then
inline** (the inlined bodies are now small), **then fuse again** across the
former call boundaries.

## Architecture Overview

The change touches 4 files (plus BUILD and test files):

```
xla/service/cpu/cpu_options.h          # Add config flag
xla/service/cpu/cpu_options.cc         # Add config flag implementation
xla/service/cpu/subcomputation_fusion.h   # New HLO pass (Phase 1)
xla/service/cpu/subcomputation_fusion.cc  # New HLO pass implementation
xla/service/cpu/cpu_compiler.cc        # Wire Phase 1 + Phase 2 into pipeline
xla/service/cpu/BUILD                  # Build targets
xla/service/cpu/subcomputation_fusion_test.cc  # Unit tests
```

The feature is gated behind an opt-in flag:
`--xla_backend_extra_options=xla_cpu_enable_two_phase_fusion`

---

## Step-by-Step Implementation

### Step 1: Add the Configuration Flag

#### File: `xla/service/cpu/cpu_options.h`

Add a new string constant and function declaration. Insert these right after the
existing `kDisableTiledEmitter` constant and `EnableTiledEmitter` declaration:

```cpp
// After this existing line:
//   inline constexpr absl::string_view kDisableTiledEmitter =
//       "xla_cpu_disable_tiled_emitter";
// Add:
inline constexpr absl::string_view kEnableTwoPhaseFusion =
    "xla_cpu_enable_two_phase_fusion";
```

And add the function declaration in the same block as the other bool functions:

```cpp
// After this existing line:
//   bool EnableTiledEmitter(const HloModuleConfig& config);
// Add:
bool EnableTwoPhaseFusion(const HloModuleConfig& config);
```

#### File: `xla/service/cpu/cpu_options.cc`

Add the implementation at the end, right before the closing namespace brace:

```cpp
// After the existing EnableTiledEmitter function, add:
bool EnableTwoPhaseFusion(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kEnableTwoPhaseFusion) > 0;
}
```

This follows the exact same pattern as every other option in this file. The flag
is read from `xla_backend_extra_options`, which is a `map<string, string>` in
the debug options proto. Presence of the key (regardless of value) enables the
feature.

---

### Step 2: Create the SubcomputationFusionPass

This is the new HLO pass that implements Phase 1.

#### File: `xla/service/cpu/subcomputation_fusion.h`

Create this file with the full Apache 2.0 license header (copy from any
adjacent file), then:

```cpp
#ifndef XLA_SERVICE_CPU_SUBCOMPUTATION_FUSION_H_
#define XLA_SERVICE_CPU_SUBCOMPUTATION_FUSION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::cpu {

// Runs CpuInstructionFusion (and optionally CpuMultiOutputFusion) on each
// non-entry sub-computation independently, in bottom-up callgraph order
// (callees before callers). This pre-compacts sub-computations so that
// subsequent full inlining + fusion operates on a smaller graph.
//
// This is Phase 1 of the two-phase fusion design: fuse within each
// sub-computation first, then inline, then fuse again across boundaries.
// FusionWrapper is intentionally deferred to Phase 3 so that Phase 3 can
// merge raw fusions rather than trying to merge already-wrapped ops.
class SubcomputationFusionPass : public HloModulePass {
 public:
  explicit SubcomputationFusionPass(bool use_multi_output_fusion)
      : use_multi_output_fusion_(use_multi_output_fusion) {}

  absl::string_view name() const override {
    return "subcomputation-fusion";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;

 private:
  bool use_multi_output_fusion_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_SUBCOMPUTATION_FUSION_H_
```

**Key design decisions:**

1. The class extends `HloModulePass` (not `HloModuleGroupPass`). This is the
   standard base class for passes that transform a single HLO module.

2. The constructor takes `use_multi_output_fusion` so the caller (cpu_compiler)
   can control whether multi-output fusion is also applied during Phase 1.

3. `using HloPassInterface::Run;` is required to avoid hiding the base class
   overload. Without this you get a compilation error.

#### File: `xla/service/cpu/subcomputation_fusion.cc`

```cpp
#include "xla/service/cpu/subcomputation_fusion.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_graph.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/service/cpu/cpu_multi_output_fusion.h"

namespace xla::cpu {

absl::StatusOr<bool> SubcomputationFusionPass::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Build the call graph to check for kCall target sub-computations.
  auto call_graph = CallGraph::Build(module, execution_threads);

  // Check if there are any kCall target sub-computations worth pre-fusing.
  bool has_call_targets = false;
  for (const CallGraphNode& node : call_graph->nodes()) {
    HloComputation* comp = node.computation();
    if (comp->IsEntryComputation() || comp->IsFusionComputation()) {
      continue;
    }
    for (const CallSite& caller_site : node.caller_callsites()) {
      if (caller_site.instruction()->opcode() == HloOpcode::kCall) {
        has_call_targets = true;
        break;
      }
    }
    if (has_call_targets) break;
  }

  if (!has_call_targets) {
    return false;
  }

  VLOG(1) << "SubcomputationFusionPass: pre-fusing sub-computations";

  // Run CpuInstructionFusion on the full module. It processes all non-fusion
  // computations via GetNonFusionComputations(), which includes the outlined
  // sub-computations. This compacts them before inlining.
  //
  // FusionWrapper is intentionally NOT run here -- it wraps remaining unfused
  // ops into single-op fusions for the emitter. Deferring it to Phase 3
  // allows Phase 3's CpuInstructionFusion to merge raw ops across former
  // call boundaries rather than trying to merge already-wrapped fusions.
  bool changed = false;
  AliasInfo alias_info;

  {
    CpuInstructionFusion instruction_fusion(
        &alias_info, /*may_duplicate=*/!use_multi_output_fusion_);
    TF_ASSIGN_OR_RETURN(bool fusion_changed,
                        instruction_fusion.Run(module, execution_threads));
    changed |= fusion_changed;
  }

  if (use_multi_output_fusion_) {
    CpuMultiOutputFusion multi_output_fusion(&alias_info);
    TF_ASSIGN_OR_RETURN(bool mof_changed,
                        multi_output_fusion.Run(module, execution_threads));
    changed |= mof_changed;

    TupleSimplifier tuple_simplifier;
    TF_ASSIGN_OR_RETURN(bool ts_changed,
                        tuple_simplifier.Run(module, execution_threads));
    changed |= ts_changed;
  }

  VLOG(1) << "SubcomputationFusionPass: "
          << (changed ? "changed" : "no change");
  return changed;
}

}  // namespace xla::cpu
```

**Critical implementation details:**

1. **Early exit if no kCall targets**: The pass first walks the call graph
   to check if there are any `kCall` instructions targeting sub-computations.
   If not (e.g., the only sub-computations are reduce bodies, which use
   `kWhile`/`kMap`/etc.), the pass returns `false` immediately. This avoids
   running the expensive fusion pass unnecessarily on modules that don't
   have outlined sub-computations.

2. **Why we use the call graph**: Sub-computations can be referenced by
   different call types. A `reduce` instruction references a body computation
   but via `kReduce`, not `kCall`. We only care about `kCall` targets because
   those are the ones that `CallInliner` will inline later. The call graph
   API (`CallGraph::Build`, `CallGraphNode`, `CallSite`) is in
   `xla/service/call_graph.h`.

3. **CpuInstructionFusion runs on ALL computations**: We don't need to
   manually iterate sub-computations. `CpuInstructionFusion` internally calls
   `GetNonFusionComputations()` which returns all non-fusion computations in
   the module, including the outlined sub-computations. So we just run it
   on the whole module and it handles everything.

4. **AliasInfo is local**: We create a fresh `AliasInfo` for Phase 1.
   Phase 3 in `cpu_compiler.cc` creates its own separate `AliasInfo`.
   This is fine because alias analysis is per-pass.

5. **FusionWrapper is NOT run in Phase 1**: This is critical. `FusionWrapper`
   wraps every remaining unfused op into a single-op fusion (for the emitter).
   If we ran it in Phase 1, then after inlining in Phase 2, Phase 3's fusion
   pass would see already-wrapped single-op fusions instead of raw ops, and
   would fail to merge them across former call boundaries. By deferring
   `FusionWrapper` to Phase 3 (where it already runs), Phase 3's
   `CpuInstructionFusion` can discover cross-boundary fusion opportunities
   with raw ops.

6. **TupleSimplifier after multi-output fusion**: Multi-output fusion can
   create tuple operations that are then redundant. `TupleSimplifier` cleans
   these up. This follows the same pattern used in the Phase 3 code in
   `cpu_compiler.cc`.

7. **may_duplicate flag**: When multi-output fusion is enabled,
   `CpuInstructionFusion` is told `may_duplicate=false` to avoid duplicating
   ops that multi-output fusion might later merge. This matches the existing
   Phase 3 behavior.

---

### Step 3: Wire the Pass into the Compiler Pipeline

#### File: `xla/service/cpu/cpu_compiler.cc`

This is the most delicate part. You need to modify the `RunHloPassesAfterLayoutAssn`
function. Find the section that adds `CpuInstructionFusion` to the pipeline.

**Add the include** (alphabetical order among the cpu/ includes):

```cpp
// After:  #include "xla/service/cpu/fusion_wrapper.h"
// Before: #include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/subcomputation_fusion.h"
```

**Add the two-phase fusion logic.** Find this existing code block:

```cpp
  AliasInfo alias_info;
  bool use_multi_output_fusion =
      options::UseMultiOutputFusion(module->config());
  pipeline.AddPass<CpuInstructionFusion>(
      &alias_info,
      /*may_duplicate=*/!use_multi_output_fusion);
```

Replace it with:

```cpp
  AliasInfo alias_info;
  bool use_multi_output_fusion =
      options::UseMultiOutputFusion(module->config());
  bool use_two_phase_fusion =
      options::EnableTwoPhaseFusion(module->config());

  if (use_two_phase_fusion) {
    // Two-phase fusion pipeline:
    //   Phase 1: Run fusion on all computations (including outlined
    //            sub-computations) to pre-compact them.
    //   Phase 2: Inline ALL sub-computations (not just single-call-site).
    //            Each inlined body is already compacted by Phase 1.
    //   Phase 3: Run fusion again on the fully inlined graph to discover
    //            cross-boundary fusion opportunities.
    // Phase 1: Run CpuInstructionFusion (+ multi-output fusion) on all
    // computations including outlined sub-computations. FusionWrapper is
    // deferred to Phase 3 so that Phase 3 can merge raw ops across
    // former call boundaries.
    pipeline.AddPass<SubcomputationFusionPass>(use_multi_output_fusion);

    // Phase 2: Inline all sub-computations (including multi-call-site).
    pipeline.AddPass<FlattenCallGraph>();
    pipeline.AddPass<CallInliner>(/*single_call_site=*/false);
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  }

  // Phase 3 (or standard single-phase): Run fusion on the (now-inlined) graph.
  pipeline.AddPass<CpuInstructionFusion>(
      &alias_info,
      /*may_duplicate=*/!use_multi_output_fusion);
```

**Key: `FlattenCallGraph` and `CallInliner` are already imported** in
`cpu_compiler.cc`. They are existing passes used elsewhere (search for
`FlattenCallGraph` and `CallInliner` in the file). `HloCSE` is also already
imported. You do NOT need to add any new includes for Phase 2.

**Modify the existing flatten-after-fusion block.** Find this existing code:

```cpp
  if (flatten_after_fusion) {
    pipeline.AddPass<FlattenCallGraph>();
    pipeline.AddPass<CallInliner>(/*single_call_site=*/true);
  }
```

Change it to:

```cpp
  if (flatten_after_fusion && !use_two_phase_fusion) {
    // Skip this when two-phase fusion is enabled since we already inlined
    // everything in Phase 2.
    pipeline.AddPass<FlattenCallGraph>();
    pipeline.AddPass<CallInliner>(/*single_call_site=*/true);
  }
```

**Why**: The existing `flatten_after_fusion` code does single-call-site inlining
after fusion. When two-phase fusion is enabled, Phase 2 already inlined ALL
sub-computations (not just single-call-site ones), so this would be redundant.
More importantly, after Phase 2 + Phase 3 there should be no sub-computations
left to inline.

**Understanding the Phase 2 passes:**

- `FlattenCallGraph()`: Makes the call graph flat by cloning computations that
  are called from multiple call sites, so each has a unique caller. This is
  required before `CallInliner` can inline multi-call-site computations.
- `CallInliner(/*single_call_site=*/false)`: Inlines ALL call instructions,
  including those whose target computation is called from multiple sites (since
  `FlattenCallGraph` already cloned them). The `false` argument means "don't
  restrict to single-call-site only".
- `HloCSE(/*is_layout_sensitive=*/true)`: Common subexpression elimination.
  After inlining, there may be duplicate sub-expressions. CSE deduplicates them.
  `is_layout_sensitive=true` because we're after layout assignment at this point.

---

### Step 4: Add BUILD Targets

#### File: `xla/service/cpu/BUILD`

Add the `cc_library` for the new pass. Place it near the other fusion-related
targets (after `fusion_wrapper` is a good spot, but order in BUILD files is
flexible):

```python
cc_library(
    name = "subcomputation_fusion",
    srcs = ["subcomputation_fusion.cc"],
    hdrs = ["subcomputation_fusion.h"],
    deps = [
        ":cpu_instruction_fusion",
        ":cpu_multi_output_fusion",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/service:call_graph",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)
```

Add the test target:

```python
xla_cc_test(
    name = "subcomputation_fusion_test",
    srcs = ["subcomputation_fusion_test.cc"],
    deps = [
        ":subcomputation_fusion",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
    ],
)
```

Add `:subcomputation_fusion` to the `deps` list of the existing `cpu_compiler`
`cc_library` target (find `name = "cpu_compiler"` in the BUILD file and add it
to its `deps` list, maintaining alphabetical order).

---

### Step 5: Write Unit Tests

#### File: `xla/service/cpu/subcomputation_fusion_test.cc`

```cpp
#include "xla/service/cpu/subcomputation_fusion.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "gtest/gtest.h"

namespace xla::cpu {
namespace {

class SubcomputationFusionTest : public HloHardwareIndependentTestBase {};

// Verifies that Phase 1 fuses elementwise ops within a sub-computation.
TEST_F(SubcomputationFusionTest, FusesWithinSubcomputation) {
  const char* hlo = R"(
    HloModule test

    helper {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      add = f32[4] add(p0, p1)
      mul = f32[4] multiply(add, p0)
      ROOT neg = f32[4] negate(mul)
    }

    ENTRY main {
      x = f32[4] parameter(0)
      y = f32[4] parameter(1)
      call1 = f32[4] call(x, y), to_apply=helper
      call2 = f32[4] call(y, x), to_apply=helper
      ROOT result = f32[4] add(call1, call2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  SubcomputationFusionPass pass(/*use_multi_output_fusion=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  EXPECT_TRUE(changed);

  // After Phase 1, the helper computation should contain a fusion node.
  const HloComputation* helper = nullptr;
  for (const auto* comp : module->computations()) {
    if (comp->name() == "helper" && !comp->IsFusionComputation()) {
      helper = comp;
      break;
    }
  }
  if (helper != nullptr) {
    bool has_fusion = false;
    for (const auto* instr : helper->instructions()) {
      if (instr->opcode() == HloOpcode::kFusion) {
        has_fusion = true;
        break;
      }
    }
    EXPECT_TRUE(has_fusion);
  }
}

// Verifies that the pass does nothing when there are no kCall sub-computations.
TEST_F(SubcomputationFusionTest, NoCallSubcomputations) {
  const char* hlo = R"(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      x = f32[100] parameter(0)
      y = f32[100] parameter(1)
      add = f32[100] add(x, y)
      mul = f32[100] multiply(add, x)
      zero = f32[] constant(0)
      ROOT result = f32[] reduce(mul, zero), dimensions={0}, to_apply=add_f32
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  SubcomputationFusionPass pass(/*use_multi_output_fusion=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  // No kCall targets exist, so the pass should early-exit without running
  // the fusion pass at all.
}

// Verifies that chained sub-computations (A calls B) are handled.
TEST_F(SubcomputationFusionTest, ChainedSubcomputations) {
  const char* hlo = R"(
    HloModule test

    inner {
      p0 = f32[4] parameter(0)
      c1 = f32[4] negate(p0)
      c2 = f32[4] negate(c1)
      ROOT c3 = f32[4] negate(c2)
    }

    outer {
      p0 = f32[4] parameter(0)
      call_inner = f32[4] call(p0), to_apply=inner
      ROOT result = f32[4] add(call_inner, p0)
    }

    ENTRY main {
      x = f32[4] parameter(0)
      call1 = f32[4] call(x), to_apply=outer
      call2 = f32[4] call(x), to_apply=outer
      ROOT result = f32[4] add(call1, call2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  SubcomputationFusionPass pass(/*use_multi_output_fusion=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));

  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace xla::cpu
```

**Testing notes:**

- `HloHardwareIndependentTestBase` provides `ParseAndReturnVerifiedModule()`
  which parses HLO text format and verifies it.
- `TF_ASSERT_OK_AND_ASSIGN` is a macro that unwraps a `StatusOr` and fails the
  test if it's an error status.
- The test for `NoCallSubcomputations` uses a `reduce` with `to_apply`, which
  creates a sub-computation but is NOT called via `kCall`. The pass should
  detect this and early-exit.

---

## How to Build and Test

```bash
# Build the pass
bazel build //xla/service/cpu:subcomputation_fusion

# Run unit tests
bazel test //xla/service/cpu:subcomputation_fusion_test

# Run the full compiler test suite to check for regressions
bazel test //xla/service/cpu:cpu_compiler_test
```

## How to Enable at Runtime

The feature is opt-in via the `xla_backend_extra_options` debug option:

```bash
# From command line (e.g., for benchmarks)
--xla_backend_extra_options=xla_cpu_enable_two_phase_fusion

# From JAX Python code
import jax
jax.config.update(
    "jax_xla_backend_extra_options",
    "xla_cpu_enable_two_phase_fusion"
)
```

## Common Pitfalls

1. **Don't run FusionWrapper in Phase 1.** FusionWrapper wraps unfused ops
   into single-op fusions for the emitter. If you do this before inlining,
   Phase 3 will see wrapped fusions instead of raw ops and won't be able to
   merge them across call boundaries.

2. **Use `CallInliner(false)` not `CallInliner(true)` in Phase 2.**
   `true` means single-call-site only, which won't inline computations called
   from multiple sites. You must use `false` (inline all) after
   `FlattenCallGraph` has cloned multi-call-site targets.

3. **Run FlattenCallGraph before CallInliner.** `CallInliner` can only inline
   a computation if it has a single call site. `FlattenCallGraph` ensures this
   by cloning shared computations.

4. **Run HloCSE after inlining.** Inlining duplicates computation bodies,
   creating redundant sub-expressions. CSE cleans these up before Phase 3
   processes them.

5. **The early-exit check matters.** Without it, you'd run the full
   `CpuInstructionFusion` pass even on modules with no outlined
   sub-computations (e.g., simple models without JAX's function outlining).
   The call graph check costs ~O(nodes) which is negligible.

6. **Guard the existing flatten_after_fusion with `!use_two_phase_fusion`.**
   Without this, the compiler would try to inline again after Phase 3, which
   is redundant (everything was already inlined in Phase 2) and could cause
   issues.

7. **`using HloPassInterface::Run;`** is needed in the header to avoid hiding
   the base class's `Run(HloModule*)` overload. Without it, calling
   `pass.Run(module.get())` without the execution_threads argument won't
   compile.

## File-by-File Diff Summary

| File | Change |
|------|--------|
| `cpu_options.h` | +2 lines: `kEnableTwoPhaseFusion` constant, `EnableTwoPhaseFusion()` declaration |
| `cpu_options.cc` | +6 lines: `EnableTwoPhaseFusion()` implementation |
| `subcomputation_fusion.h` | +57 lines: new file, class declaration |
| `subcomputation_fusion.cc` | +98 lines: new file, `Run()` implementation |
| `cpu_compiler.cc` | +29/-1 lines: include, Phase 1+2 pipeline insertion, flatten guard |
| `BUILD` | +34 lines: `subcomputation_fusion` library + test targets, dep in `cpu_compiler` |
| `subcomputation_fusion_test.cc` | +157 lines: new file, 3 test cases |
