# Whole-Program Codegen for Small XLA:CPU Programs

Branch: `claude/optimize-thunk-runtime-xWU05`

## 1. Problem & Goals

### 1.1 Symptom

Small JAX/XLA programs on xla:cpu regressed significantly in per-call runtime
after the backend switched to the **thunk runtime** (one scheduled thunk per
HLO op / fusion / library call, coordinated by `ThunkExecutor`). Prior to that
switch, the backend emitted a single LLVM module for the entire entry
computation and the per-call overhead was essentially one function call.

Two upstream-reported repros that this design targets:

- **Example A — 35-joint mock mass matrix** (JAX issue `jax-ml/jax#26021`): a
  pure-forward computation built from many small `dot`s, `reduce`s, elementwise
  ops, and `dynamic-update-slice`s. Pre-regression: fast (low tens of µs per
  call). Post-regression: multi-hundred-µs per call, dominated by thunk
  dispatch, not arithmetic.
- **Example B — Cholesky-like `lax.scan`** (JAX issue
  `jax-ml/jax#36799 comment 4270944098`): a 200-iteration scan whose body is a
  handful of small ops (`sqrt`, two small matmuls, `outer`). Pre-regression:
  tens of µs per scan. Post-regression: much slower because each iteration
  dispatches many thunks through `WhileThunk`.

### 1.2 Root cause

Thunk dispatch has fixed per-thunk overhead (executor bookkeeping, buffer
resolution, task-queue interaction, potential hop into `Eigen::ThreadPool`).
For small programs this overhead dwarfs the actual compute. `ThunkExecutor`
already has `execute_sequential_*_threshold` knobs
(`xla/backends/cpu/runtime/thunk_executor.h:62–67`) that short-circuit to
sequential execution, but they only skip the concurrency layer — the per-thunk
cost is still paid N times.

The fundamental fix is to stop emitting N thunks for programs that could
instead be compiled into one LLVM function invoked by a single
`KernelThunk`. This recovers the pre-thunk-runtime dispatch profile.

### 1.3 Goals (v1)

1. **≥ 2× per-call speedup** on Example A (pure forward; full program becomes
   one kernel) and Example B (scan body becomes one kernel; `WhileThunk` still
   drives iteration).
2. **Zero regression** on large-model workloads (hand-tuned dot/conv/collective
   calls to libraries must still happen). Any program the heuristic declines
   falls through to today's behavior byte-for-byte.
3. **Use the fusion-emitter path** (`xla/backends/cpu/codegen/emitters/`,
   `FusionCompiler`) — not `ir_emitter` / `ir_emitter2`. The mechanism is to
   rewrite a region of the HLO into a single `kFusion` instruction that the
   existing fusion-emitter pipeline already knows how to lower.
4. **Flag-gated** with a sentinel that disables the feature entirely, so we
   can ship defensively and bisect.

### 1.4 Success criteria

| Metric                                    | Target                          |
|-------------------------------------------|---------------------------------|
| Example A avg per-call time               | ≥ 2× faster than current main   |
| Example B median per-call time            | ≥ 2× faster than current main   |
| Existing XLA:CPU test suite               | All green                       |
| Large model benchmarks (TBD picklist)     | Within noise of current main    |
| Thunk count for Example A after pass      | 1 `KernelThunk` for entry comp. |
| Thunk count for Example B body            | 1 `KernelThunk` inside `WhileThunk` body |

## 2. Non-Goals (v1 Scope Fence)

The following are explicitly **out of scope for v1**. Each has a v2 entry in
§12 if relevant.

1. **Inlining `while` / `conditional` / `call` into the mega-fusion.** v1
   treats these as hard region boundaries: the pass merges inside each
   computation (including called computations like while-bodies), but never
   folds the control-flow instruction itself into a `kFusion`. This keeps
   `WhileThunk` / `ConditionalThunk` driving iteration/selection — which is
   enough to recover Example B because the scan *body* becomes one kernel.
2. **Absorbing runtime-coordinated ops** of any kind: collectives
   (`all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`,
   `collective-permute`), `infeed`, `outfeed`, `custom-call`,
   `rng-get-and-update-state`, `partition-id`, `replica-id`. These always stay
   as their own thunks and act as region barriers.
3. **Replacing existing fusion passes.** `CpuInstructionFusion`,
   `FusionWrapper`, and (optionally) `CpuMultiOutputFusion`
   (`cpu_compiler.cc:1053–1068`) keep running untouched. The new pass runs
   **after** them and merges the `kFusion` instructions they produced — plus
   a small set of non-fusion ops the cost model says are cheap enough to
   absorb.
4. **Library-path changes.** The dot/conv dispatch picker in
   `dot_op_emitter.cc:1411` and the oneDNN/YNN rewriters are untouched. A
   `dot` already rewritten to a library call (e.g., a `__onednn$matmul`
   custom-call) is a barrier. Only dots that would otherwise be codegenned
   are candidates for absorption, and the per-op cost model decides.
5. **New cost model for parallelism.** v1 ships the simplest useful model
   (kflops estimate + a coarse "parallelizable?" bit per op). No tiling
   analysis, no hardware introspection, no learned model.
6. **Changes to `ThunkExecutor` or to the thunk runtime itself.** The fix is
   entirely at HLO-to-MLIR compile time. No changes to
   `xla/backends/cpu/runtime/`.
7. **Changes to `IrEmitter` / `IrEmitter2`.** The mega-fusion is emitted
   through the MLIR fusion-emitter pipeline (`FusionCompiler`) exclusively.
   If `FusionCompiler` cannot compile a particular op today, that op is not
   eligible for absorption in v1.
8. **Multi-threaded execution of a mega-fusion.** The emitted kernel runs on
   one worker. The cost model must therefore decline to absorb ops whose
   parallel thunk dispatch is a net win (high kflops + high parallelism).
9. **GPU backend, TPU backend, or any non-CPU target.** Whole-program codegen
   for GPU is a separate effort; this document is xla:cpu-only.
10. **Ahead-of-time / serialized-executable path regressions.** The pass must
    be idempotent and deterministic so AOT artifacts remain stable, but we
    do not expand AOT-specific functionality in v1.

## 3. High-Level Architecture

The design adds **one new HLO pass** (`MegaFusionPass`) and **one small
helper** (`CpuCostModel`), plus a flag to gate both. All other XLA:CPU code
paths are unchanged.

### 3.1 Where it sits in the pipeline

`CpuCompiler::RunHloPasses` (`cpu_compiler.cc:~1050`) today ends its fusion
stage with, in order:

```
CpuInstructionFusion         (forms kFusions from elementwise/reduce chains)
FusionWrapper                (wraps stray codegennable ops as 1-op kFusions)
CpuMultiOutputFusion?        (optional, merges fusions sharing operands)
FlattenCallGraph? etc.
```

After this stage, every codegen-eligible instruction is already a
`kFusion`. We insert `MegaFusionPass` **immediately after** this fusion
stage and **before** scheduling / buffer assignment. By that point the HLO
is in a clean "kFusions + library-call custom-calls + control flow +
collectives" form, which is exactly the shape `MegaFusionPass` is designed
to walk.

```
  … existing passes …
  CpuInstructionFusion
  FusionWrapper
  [CpuMultiOutputFusion]
+ MegaFusionPass   ← new
  … scheduling, buffer assignment, thunk emission …
```

### 3.2 What the pass does, in one paragraph

For each computation in the module (entry computation **and** called
computations such as while-bodies), `MegaFusionPass` walks instructions in
a deterministic order and greedily grows contiguous **regions** of
absorbable ops. An op is absorbable if the cost model says it is either
(a) low-kflops or (b) high-kflops but poorly parallelizable, **and** it is
not a runtime-coordinated op (§2 item 2), **and** it is not already a
library call. When a region accumulates more than `kflops_threshold`
kflops, or hits a barrier (control flow, collective, library call,
infeed/outfeed, etc.), the region is closed. Each closed region containing
≥ 2 absorbable ops is rewritten into a single `kFusion` instruction by
outlining the region's ops into a fresh fused computation, with
tuple-typed output covering every value the region produces that is
consumed outside. Single-op regions are left alone (they are already
kFusions from `FusionWrapper`).

### 3.3 What the rest of the stack sees

After `MegaFusionPass`, the HLO is a strictly valid XLA module: the new
`kFusion`s are indistinguishable in kind from ones produced by
`CpuInstructionFusion`. The scheduler schedules them normally, buffer
assignment assigns buffers normally, and **the existing fusion-emitter
path (`FusionCompiler`, `cpu_fusion_emitter.cc`) lowers them to LLVM
normally.** `ThunkEmitter` emits one `KernelThunk` per mega-fusion, just
as it would for any other fusion. No thunk-runtime changes are required.

### 3.4 Why a kFusion (not a new instruction kind / new thunk type)

Three reasons:

1. **Reuse the lowering pipeline.** `FusionCompiler` already handles
   tuple-output fusions, bufferization, tiling, and LLVM codegen. Anything
   we express as a `kFusion` inherits all of it for free.
2. **Reuse downstream infra.** Buffer assignment, alias analysis, liveness,
   `ThunkEmitter::EmitFusionKernelThunk`, `KernelThunk` — none of these
   need to know about "mega" vs "regular" fusions.
3. **Fail-safe.** If `FusionCompiler` can't lower a specific mega-fusion
   (e.g., an absorbed op uses an emitter feature not yet supported), the
   absorption predicate's "FusionCompiler can lower this op?" check is the
   single source of truth. A missed check produces a clean compile error
   that a targeted test can catch — not silent wrong output.

### 3.5 Cost-model role, in one paragraph

`CpuCostModel` is a pure function from an `HloInstruction*` to
`(kflops_estimate, parallelism_score)`. v1 implements it with small
op-specific closed-form estimates (FLOPs from `HloCostAnalysis` or a tiny
table; parallelism score from output-shape element count and op kind).
`MegaFusionPass` calls it once per candidate op. The **flag threshold is
expressed in kflops**, so the cost model's units are exposed directly to
users. The parallelism score is only used internally to decide whether a
high-kflops op should still be absorbed (unparallelizable) or left as its
own thunk (parallelizable).

### 3.6 Dataflow summary

```
HLO module
   │
   ▼  (existing passes, unchanged)
HLO w/ kFusions + library custom-calls + control flow + collectives
   │
   ▼  MegaFusionPass       ← this design
HLO w/ FEWER kFusions, each possibly much larger
   │
   ▼  scheduling, buffer assignment
   ▼  ThunkEmitter
Thunk sequence: fewer KernelThunks, same library/collective thunks
   │
   ▼  FusionCompiler lowers each kFusion to LLVM
Executable
```

## 4. Component Specs

### 4.1 CPU Cost Model (kflops + parallelism)

**New file:** `xla/service/cpu/cpu_cost_model.{h,cc}`

**Purpose:** supply `MegaFusionPass` with a cheap, deterministic per-op
estimate of (work, parallelism). Not a general-purpose cost model — just
enough signal to gate absorption.

**Public API:**

```c++
namespace xla::cpu {

struct OpCost {
  // Estimated floating-point (or equivalent integer) work, in kflops.
  // 0 for bitcast-like ops (reshape, transpose that is a layout no-op,
  // broadcast that lowers to a stride change).
  double kflops = 0.0;

  // Rough count of independent output work items. Higher => op benefits
  // more from multi-threaded execution. For elementwise ops this is the
  // output element count; for reductions it's the *output* (not input)
  // count; for dots it's the output element count (M*K for MxN @ NxK).
  int64_t parallelism_score = 0;
};

class CpuCostModel {
 public:
  // v1: stateless. Future versions may hold HloCostAnalysis or
  // TargetMachineFeatures.
  CpuCostModel() = default;

  // Returns OpCost for a single HLO instruction. Never fails; returns
  // a conservative {0 kflops, INT64_MAX parallelism} for instructions
  // the model doesn't know how to estimate (so they stay *out* of
  // mega-fusions by default — see §5.3).
  OpCost Estimate(const HloInstruction* instr) const;

  // Convenience: true iff `instr` is considered "highly parallelizable"
  // for the purpose of absorption decisions.
  bool IsParallelWin(const HloInstruction* instr) const;

 private:
  // Threshold above which parallelism_score counts as a "parallel win".
  // v1 constant; later may become target-aware.
  static constexpr int64_t kParallelWinThreshold = 4096;
};

}  // namespace xla::cpu
```

**Estimation rules (v1):**

| HLO opcode                                | kflops formula                | parallelism_score |
|-------------------------------------------|-------------------------------|-------------------|
| elementwise unary / binary / ternary      | `out_elems * 1 / 1000`        | `out_elems`       |
| `dot` (contraction `K`, output `M*N`)     | `2 * M * N * K / 1000`        | `M * N`           |
| `reduce` (input `IE`, output `OE`)        | `IE / 1000`                   | `OE`              |
| `convolution`                             | (reuse `HloCostAnalysis`)     | output_elems      |
| `broadcast`, `reshape`, `transpose`, `bitcast` | `0`                      | `out_elems`       |
| `dynamic-slice`, `dynamic-update-slice`   | `out_elems / 1000`            | `out_elems`       |
| `concatenate`, `pad`                      | `out_elems / 1000`            | `out_elems`       |
| `iota`, `constant`                        | `0`                           | `out_elems`       |
| `tuple`, `get-tuple-element`              | `0`                           | `INT64_MAX` (free)|
| unknown opcode                            | `0` kflops, `INT64_MAX` score | (treated as "parallel win" ⇒ not absorbed) |

Rationale for unknowns: returning `INT64_MAX` parallelism means
`IsParallelWin` is `true`, and §5.3's absorption predicate rejects
unknowns unless an explicit allow-list check passes. The effect is
fail-safe: new opcodes don't accidentally get absorbed.

For `dot` specifically: after the library rewrite passes
(`LibraryRewriter`, `onednn_contraction_rewriter`), any `dot` still
present as a `dot` instruction is a codegen-path dot. Those are the ones
we care about gating. Dots rewritten to `__onednn$…` custom-calls are
barriers and never reach the predicate.

### 4.2 MegaFusionPass (HLO pass)

**New file:** `xla/service/cpu/mega_fusion_pass.{h,cc}`

**Class outline:**

```c++
namespace xla::cpu {

class MegaFusionPass : public HloModulePass {
 public:
  struct Options {
    // Close a region when its accumulated kflops exceeds this. 0 disables
    // the pass entirely (the pass's Run() becomes a no-op).
    double kflops_threshold = 0.0;

    // Minimum region size to emit as a mega-fusion. Regions of 1
    // absorbable op are skipped (they're already singleton kFusions
    // from FusionWrapper).
    int min_region_size = 2;
  };

  MegaFusionPass(Options options, const CpuCostModel* cost_model);

  absl::string_view name() const override { return "mega-fusion"; }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;

 private:
  // Walks one computation; returns true if the computation was modified.
  absl::StatusOr<bool> RewriteComputation(HloComputation* computation);

  // Given a closed region (vector of instructions in schedule order),
  // rewrite it into a single kFusion instruction, preserving all
  // external uses via a tuple output. See §5.4.
  absl::Status EmitRegionAsFusion(
      HloComputation* computation,
      absl::Span<HloInstruction* const> region);

  const Options options_;
  const CpuCostModel* cost_model_;  // not owned
};

}  // namespace xla::cpu
```

Semantics:

- If `options_.kflops_threshold <= 0`, `Run` returns `false` immediately
  (no-op). This is the **disable sentinel**.
- Otherwise, iterates `module->MakeNonfusionComputations()` (so we walk
  entry + while-bodies + conditional-branches + call-targets, but not
  the insides of existing kFusions — those are leaves). See §5.1 for
  the per-computation gate.
- Deterministic: iteration order of computations and instructions is
  stable; no hashing of pointers.

### 4.3 Flag plumbing

**Files touched:**

- `xla/service/cpu/cpu_options.h` — add constant + accessor.
- `xla/service/cpu/cpu_options.cc` — accessor implementation.
- `xla/xla.proto` (or wherever `DebugOptions` lives) — add field
  `int64 xla_cpu_whole_program_kflops_threshold = N;` with a doc comment
  explaining `0 = disabled` and the default.

**Constant & accessor:**

```c++
// cpu_options.h
inline constexpr absl::string_view kWholeProgramKflopsThreshold =
    "xla_cpu_whole_program_kflops_threshold";

// Returns the kflops threshold; 0 (the sentinel) disables MegaFusionPass.
int64_t WholeProgramKflopsThreshold(const HloModuleConfig& config);
```

**Default:** `50` kflops (tentative; §11 open question). Small enough to
cover both example models end-to-end, large enough that any dense
BLAS-scale dot (say, 1024×1024 × 1024×1024 = ~2.1M kflops) blows past it
and stays as a separate thunk.

**Sentinel:** any value `<= 0` disables the pass. This is both the
off-switch and the bisect knob. We document `0` as the canonical
"disabled" value.

### 4.4 Pipeline integration

**File touched:** `xla/service/cpu/cpu_compiler.cc` around line 1068 —
immediately after the existing fusion stage (after `FusionWrapper` and
the optional `CpuMultiOutputFusion`).

```c++
// cpu_compiler.cc, after the existing fusion block (~line 1068)
const int64_t mega_kflops =
    options::WholeProgramKflopsThreshold(module->config());
if (mega_kflops > 0) {
  pipeline.AddPass<MegaFusionPass>(
      MegaFusionPass::Options{
          /*kflops_threshold=*/static_cast<double>(mega_kflops),
          /*min_region_size=*/2},
      cost_model_.get());
}
```

Notes on placement:

- **After** `FusionWrapper`: guarantees that every codegennable op we
  might absorb is already a `kFusion`, which makes the outlining logic
  in §5.4 much simpler (we mostly splice existing fused computations
  rather than extract scalars).
- **Before** scheduling / buffer assignment: mega-fusions become single
  schedule nodes, so they get coherent buffer allocation and sit on the
  critical path naturally.
- **Before** the collective-combiner and other late passes: those passes
  walk collectives and don't interact with fusions, so ordering is
  free; we pick "immediately after fusion" for locality of reasoning.

The `cost_model_` owner: `CpuCompiler` gains a
`std::unique_ptr<CpuCostModel> cost_model_` member, constructed once in
`CompileCpuExecutable` and threaded into the pipeline. This matches the
existing `AliasInfo` ownership pattern on the surrounding lines.

## 5. Detailed Algorithm: MegaFusionPass

The pass is one `Run` method. Everything below is inside its body (or
helpers it calls). Pseudocode is kept close to the real C++ so a reader
can map it line-by-line to the implementation.

### 5.1 Preconditions & per-computation gate

```
Run(module, execution_threads):
  if options_.kflops_threshold <= 0:              # disable sentinel
    return false
  changed = false
  for computation in module->MakeNonfusionComputations(execution_threads):
    if !IsEligibleComputation(computation):
      continue
    changed |= RewriteComputation(computation)
  return changed
```

**`IsEligibleComputation(c)` returns false when:**

1. `c->IsFusionComputation()` — already a fusion body; treated as a leaf
   by the outer walk. (Redundant with `MakeNonfusionComputations` but
   makes the intent explicit.)
2. `c->IsAsyncComputation()` — async-launched bodies stay as-is in v1.
3. `c` is a `custom-call` target body — same reasoning.
4. `c` has ≤ 1 non-parameter, non-root-tuple instructions — nothing to
   merge.

Note: while-bodies and conditional-branches **are** eligible (this is
how Example B benefits in v1). The `WhileInstruction` /
`ConditionalInstruction` in the *calling* computation stays a barrier;
but the body's instructions are walked and mega-fused.

### 5.2 Schedule-order region growing

Within one computation, the pass walks instructions in
`computation->MakeInstructionPostOrder()` order (a stable topological
order). It maintains a mutable `Region` struct:

```c++
struct Region {
  std::vector<HloInstruction*> members;   // in topological order
  double kflops_total = 0.0;
};
```

Algorithm (pseudocode):

```
RewriteComputation(c):
  changed = false
  region = empty
  for instr in c->MakeInstructionPostOrder():
    if IsBarrier(instr):
      changed |= Close(c, region)
      region = empty
      continue                                   # skip barrier itself
    if !IsAbsorbable(instr):
      changed |= Close(c, region)
      region = empty
      continue                                   # unknown/ineligible
    cost = cost_model_->Estimate(instr)
    # Would adding this op blow the threshold?
    if !region.empty() and
       region.kflops_total + cost.kflops > options_.kflops_threshold:
      changed |= Close(c, region)
      region = empty
    region.members.push_back(instr)
    region.kflops_total += cost.kflops
  changed |= Close(c, region)                    # flush tail
  return changed

Close(c, region):
  if region.members.size() < options_.min_region_size:
    return false                                 # size 0 or 1: skip
  EmitRegionAsFusion(c, region.members)
  return true
```

**Why post-order / topological order is correct.** Regions are grown by
appending only; they never re-enter after being closed. Because we
process instructions in a topological order, every operand of an
absorbable instruction `I` is produced either (a) before the current
region began (external operand — becomes a kFusion parameter), or (b)
inside the current region (internal dataflow). No edge can flow
backwards into a closed region.

**Cycle / ordering invariant.** After rewriting, the new kFusion sits
in the computation where its last member used to sit. Every consumer
of a former region output now consumes from the kFusion root (or a
`get-tuple-element` of it). Because all those consumers were
topologically *after* every region member, the new kFusion is still
topologically valid.

**Single-pass.** v1 runs one sweep per computation. A second sweep
could merge mega-fusions across barriers that are now mergeable
themselves, but in practice barriers are fixed (control flow,
collectives, library dots) — a second sweep adds nothing for the
shapes we care about. Mark as a v2 consideration (§12).

### 5.3 Absorption predicate (per-op)

`IsBarrier(instr)` — **hard stops**. Any of these ends the region:

- `kWhile`, `kConditional`, `kCall` — v1 non-goal (§2.1).
- All collectives: `kAllReduce`, `kAllGather`, `kReduceScatter`,
  `kAllToAll`, `kCollectivePermute`, `kAllReduceStart`/`kDone`, etc.
- `kInfeed`, `kOutfeed`.
- `kCustomCall` — includes library dots/convs rewritten by
  `LibraryRewriter` and by `onednn_contraction_rewriter`.
- `kSend`, `kRecv`, `kSendDone`, `kRecvDone`.
- `kRngGetAndUpdateState`, `kPartitionId`, `kReplicaId`.
- `kAfterAll`, `kAddDependency` (ordering-only; safest to be barriers).
- `kTrace`.

`IsAbsorbable(instr)` — must be **true** for inclusion:

1. `instr->opcode() != kParameter && instr->opcode() != kConstant` —
   parameters and constants don't need to be "absorbed"; they flow in
   as operands of the fused region naturally. (Constants inside a
   region become constants inside the fused computation per HLO's
   fusion mechanics.)
2. `instr != instr->parent()->root_instruction()` — the root is never
   an independent absorption target; a region whose *last member* is
   the root is fine (§5.4 handles this).
3. `instr->opcode() == kFusion` — always absorbable (these are the
   common case after `FusionWrapper`).
4. Otherwise: `CpuFusionEmitterSupportsOpcode(instr->opcode())` — a
   small allow-list matching what `cpu_fusion_emitter.cc` can lower
   today. v1 list: all elementwise opcodes, `reduce`, `reshape`,
   `transpose`, `broadcast`, `iota`, `dot`, `dynamic-slice`,
   `dynamic-update-slice`, `concatenate`, `pad`, `slice`, `gather`
   (only if already fusion-emitter-supported — double-check against
   `cpu_fusion_emitter_test.cc` coverage).
5. **Parallelism gate for expensive non-fusion ops.** If
   `cost.kflops > options_.kflops_threshold / 4` (a quarter of the
   region budget) **and** `cost_model_->IsParallelWin(instr)` is
   `true`, the op is **not** absorbed. Rationale: a single expensive
   parallelizable op (e.g., a medium-sized `dot`) is a better thunk —
   the threaded library dispatch amortizes its own overhead. This is
   the concrete realization of answer #4 from the spec Q&A.
6. All operands of `instr` must be reachable without crossing a
   barrier. With topological-order region growth, this is automatic
   **except** when an operand's producer was skipped (barrier) and
   the value was routed around. The check is: for every operand
   `op`, `op` is either (a) outside the current region's `members`
   set *and* produced before the region began, or (b) inside the
   region. This is O(1) per operand with a small hash set.

Any op that is neither a barrier nor absorbable (rule 4 fails, rule 5
fails, etc.) is a **soft barrier**: it closes the current region but
is itself left untouched in the HLO, and the next absorbable op
starts a new region.

### 5.4 Emitting the kFusion instruction

`EmitRegionAsFusion(c, members)` is where the rewrite happens.

**Path A — all members are already `kFusion`s (common case after
`FusionWrapper`).** Use XLA's existing fusion-merging primitive:

```c++
// Starting from the last member, repeatedly fuse predecessors into it.
HloInstruction* accumulator = members.back();
for (auto it = members.rbegin() + 1; it != members.rend(); ++it) {
  HloInstruction* earlier = *it;
  // MergeFusionInstructionIntoMultiOutput handles:
  //   - inlining `earlier`'s fused computation into accumulator's
  //   - rewiring earlier's users to accumulator's matching output
  //   - deleting `earlier`
  TF_RETURN_IF_ERROR(
      accumulator->MergeFusionInstructionIntoMultiOutput(
          Cast<HloFusionInstruction>(earlier)));
}
```

This is the same primitive `CpuMultiOutputFusion` uses today; the
difference is that we pick adjacency in **schedule order** rather than
by shared-operand heuristics. Reuse means no new HLO-manipulation code.

**Path B — region contains a non-fusion absorbable op** (e.g., a small
`dot` we chose to absorb). Promote it to a singleton fusion first,
then fall into Path A:

```c++
for (HloInstruction*& m : members) {
  if (m->opcode() != HloOpcode::kFusion) {
    TF_ASSIGN_OR_RETURN(
        m, c->CreateFusionInstruction({m}, HloInstruction::FusionKind::kLoop));
  }
}
// …then Path A.
```

`CreateFusionInstruction` rewires operands/users correctly. After this
loop every `m` is a `kFusion` and Path A applies unchanged.

**FusionKind.** The merged accumulator inherits its kind from its
rightmost input (typically `kLoop`). If any member has
`FusionKind::kInput` (reduction-style), promote the accumulator to
`kInput`. v1 rule: `kInput` if any member is `kInput`, else `kLoop`.
This matches existing CPU fusion-emitter expectations.

### 5.5 Tuple/multi-output handling

After merging, the accumulator kFusion may have multiple outputs (one
per original region output that is consumed outside the region). Two
cases:

1. **Single external consumer pattern.** All region outputs flow into
   one successor op. `MergeFusionInstructionIntoMultiOutput` produces
   a root `kTuple` and inserts `kGetTupleElement`s. XLA handles this
   natively — no extra work.
2. **Some region outputs are the computation root.** If the root of
   `c` was inside the region, the region's mega-fusion becomes `c`'s
   new root (possibly via a top-level `kTuple` if `c`'s original root
   was a tuple). The merge primitive handles this as long as we keep
   the root instruction up to date:
   ```c++
   if (c->root_instruction() == members.back_original) {
     c->set_root_instruction(accumulator);
   }
   ```
   In practice `MergeFusionInstructionIntoMultiOutput` already rewires
   root-producing fusions when the replaced instruction had the root
   as a user; we just call it after each merge and let it do its job.

**Dead-output pruning.** After all merges, run
`accumulator->DeduplicateFusionOperands()` and a tuple-output
pruner (`HloDCE` is typically scheduled later and handles this, but we
can also call `accumulator->fused_instructions_computation()->
RemoveUnusedParametersAndElements()` defensively). v1: rely on the
existing late `HloDCE` pass; only add a targeted call if tests show
dead outputs surviving.

### 5.6 Buffer aliasing & layout constraints

**In-place ops (`dynamic-update-slice`, `scatter`).** XLA's buffer
assignment handles in-place aliasing via `AliasInfo` and per-op
fusion-emitter metadata. A `dynamic-update-slice` inside a fusion is
fully supported today. No special handling needed here.

**Layout.** `MegaFusionPass` runs **before** layout assignment (which
happens later in the pipeline). Fusions are layout-neutral at the HLO
level; layout assignment propagates through fusion operands and roots.
No new layout constraints are introduced.

**Alias analysis.** The new mega-fusion is an ordinary kFusion from
alias analysis's perspective. No changes to `AliasInfo` or to
`HloAliasAnalysis` are required. (Verify with the existing
`cpu_compiler_internals_test.cc` after implementation.)

**Copy-insertion interactions.** Copy-insertion runs after fusion
passes and inserts copies where aliasing would violate semantics.
Growing fusions may increase pressure on copy-insertion in pathological
cases. v1 mitigation: the `min_region_size = 2` guard plus the
kflops threshold keeps mega-fusions bounded. Watch for copy-count
regressions in the test plan (§9.3).

## 6. Data Structures

Consolidated reference. All types are already specified in §4–§5; this
section is a single place to eyeball them.

**Public (new headers):**

```c++
// xla/service/cpu/cpu_cost_model.h
namespace xla::cpu {

struct OpCost {
  double kflops = 0.0;            // 1e3 flop units; 0 for bitcast-like ops
  int64_t parallelism_score = 0;  // rough independent-work count
};

class CpuCostModel {
 public:
  CpuCostModel() = default;
  OpCost Estimate(const HloInstruction* instr) const;
  bool IsParallelWin(const HloInstruction* instr) const;

 private:
  static constexpr int64_t kParallelWinThreshold = 4096;
};

}  // namespace xla::cpu
```

```c++
// xla/service/cpu/mega_fusion_pass.h
namespace xla::cpu {

class MegaFusionPass : public HloModulePass {
 public:
  struct Options {
    double kflops_threshold = 0.0;  // <= 0 disables the pass
    int min_region_size = 2;
  };

  MegaFusionPass(Options options, const CpuCostModel* cost_model);
  absl::string_view name() const override { return "mega-fusion"; }
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override;

 private:
  absl::StatusOr<bool> RewriteComputation(HloComputation* computation);
  absl::Status EmitRegionAsFusion(
      HloComputation* computation,
      absl::Span<HloInstruction* const> region);

  Options options_;
  const CpuCostModel* cost_model_;  // not owned
};

}  // namespace xla::cpu
```

**Internal (inside `mega_fusion_pass.cc`, unit-of-translation only):**

```c++
namespace {

// Growing region state. Lives on the stack of RewriteComputation.
struct Region {
  std::vector<HloInstruction*> members;  // topological order
  double kflops_total = 0.0;
  // Fast lookup for operand-locality checks in IsAbsorbable.
  absl::flat_hash_set<const HloInstruction*> member_set;
};

// Allow-list of opcodes the CPU fusion emitter can lower today.
// Single source of truth — §5.3 rule 4 consults this set. Updating
// this set is the mechanism for enabling new opcodes to participate.
const absl::flat_hash_set<HloOpcode>& FusionEmitterSupportedOpcodes();

// Hard-stop opcodes per §5.3.
bool IsBarrier(const HloInstruction* instr);

// Combined predicate per §5.3.
bool IsAbsorbable(const HloInstruction* instr,
                  const Region& region,
                  const MegaFusionPass::Options& options,
                  const CpuCostModel& cost_model);

}  // namespace
```

**Flag field (proto):**

```proto
// In xla.proto, DebugOptions message:
int64 xla_cpu_whole_program_kflops_threshold = <next_unused_field_id>;
// Docstring: "Merge contiguous HLO regions whose accumulated estimated
// work stays below this many kflops into a single kFusion, producing
// one KernelThunk per region. 0 disables the optimization entirely.
// Default: 50."
```

**Flag accessor (existing header):**

```c++
// xla/service/cpu/cpu_options.h additions:
inline constexpr absl::string_view kWholeProgramKflopsThreshold =
    "xla_cpu_whole_program_kflops_threshold";
int64_t WholeProgramKflopsThreshold(const HloModuleConfig& config);
```

**CpuCompiler member (existing header):**

```c++
// xla/service/cpu/cpu_compiler.h, inside class CpuCompiler:
std::unique_ptr<CpuCostModel> cost_model_;  // constructed once per compile
```

No changes to thunk types, executor, or runtime-side structures.

## 7. Worked Examples

### 7.1 Example 1: 35-joint mass-matrix (pure forward)

**Source:** `mock_mass_matrix(q)` from the linked JAX issue (35 joints,
float64). After JAX tracing and the upstream XLA optimization passes,
the entry computation contains roughly:

- 35 iterations of: `cos`, `sin`, two `dynamic-update-slice`s, a small
  3x3 `dot` (rotation composition), a small 4x4 `dot` (homogeneous
  transform). (≈ 35 * ~10 = 350 small ops, all codegen-path.)
- 35 iterations of inertia construction: skew-symmetric builds,
  element-wise arithmetic, a few 3x3 `dot`s. (≈ 200 small ops.)
- `block`ing reshapes/concatenates to build `Rs (35x6x6)`.
- Two batched `einsum`s (`Ic = Rs @ I @ Rs.T`) — each is a batched
  3x3 `dot` pattern, 35 batches, so 35 small dots (6x6 @ 6x6). Each
  dot is ~2 * 6 * 6 * 6 ≈ 0.4 kflops.
- A final `einsum` over mask × Ic → `composite_inertias`: a 35x35
  mask × 35x6x6 tensor product. The largest single dot is roughly
  35x35 @ 35x36 = ~90 kflops.
- `M_part`, `M`, `M_lower`, symmetric completion — dozens more small ops.

**Total estimated kflops** for the entire entry computation:
well under 1,000 kflops (every individual op is small; the aggregate
is dominated by the final `einsum`s, each a few tens of kflops).

**Post-existing-fusion-pass shape.** `CpuInstructionFusion` +
`FusionWrapper` collapse most of the per-iteration arithmetic into
singleton or small multi-op `kFusion`s. We estimate ~300 kFusions,
zero barriers (no while, no collective, no library dots — the small
dots are below any library-rewrite threshold), and zero custom-calls.

**MegaFusionPass walk (threshold = 50 kflops).**

```
region = []; total = 0.0
i=0:   kFusion {cos,sin,DUS,DUS}   cost ~0.01 kflops   → push
...
i=N:   kFusion {small 6x6 dot}     cost ~0.0004 kflops → push
...
at some point total >= 50 → Close region (say, at i=K₁, ~80 fusions)
region = []; total = 0.0; resume
...
end of walk → Close final region
```

**Expected outcome.** 3–6 mega-fusions covering the entire entry
computation (exact count depends on where the 50 kflops boundary
falls among ~1,000 total kflops). Each becomes one `KernelThunk`. No
other thunks (no control flow, no collectives). **Thunk count drops
from ~300 → ~5.**

**Speedup mechanism.** Each eliminated thunk saves ~1–5 µs of dispatch
+ buffer-resolution overhead. 295 fewer thunks × ~2 µs ≈ 590 µs
saved. Baseline is hundreds of µs/call; target is tens of µs/call.
Well above the ≥2× bar.

**Fallback check.** If the cost model under-estimates aggregate work
and the threshold is tuned low (e.g., 10 kflops), regions become
smaller and we get more mega-fusions — still a big win. If tuned
high (e.g., 5,000 kflops), we get 1 mega-fusion for the whole
program — also a big win. The example is robust to threshold tuning
within a wide range; it only fails to benefit if the threshold is
essentially 0 (disabled) or if a key op slips out of the
fusion-emitter allow-list.

### 7.2 Example 2: Cholesky-like scan (while body)

**Source:** `lax.scan(body, init, (d,p,q,a))` with N=2000 iterations
and M=3, float64. `lax.scan` lowers to a `while` instruction whose
body is:

```
body(carry, data):  # carry=fp (3x3), data=(dk, pk, qk, ak)
  t0 = pk @ fp                              # 1x3 · 3x3 → 1x3   (dot)
  t1 = t0 @ pk                              # 1x3 · 3   → scalar(dot/reduce)
  ck = sqrt(dk - t1)                        # scalar elementwise
  tmp = fp @ ak.T                           # 3x3 · 3x3 → 3x3   (dot)
  w_num = qk - pk @ tmp                     # 1x3 elementwise
  wk = w_num / ck                           # 1x3 elementwise
  carry' = ak @ tmp + outer(wk, wk)         # 3x3 · 3x3 + 3x3   (dot + add)
  return carry', (ck, wk)
```

**Post-existing-fusion-pass shape of the while body.** A handful of
kFusion instructions: the three small 3x3 `dot`s (each ~0.054 kflops),
and several elementwise/reduction kFusions around them. No
collectives, no library dots (all dots are tiny). The while body is
its own `HloComputation` reachable from `MakeNonfusionComputations`.

**MegaFusionPass walk of the while body (threshold = 50 kflops).**

Total work per iteration is well under 1 kflop. The entire while
body fuses into **one** mega-kFusion.

**Walk of the entry computation (caller of while).** The entry
computation contains the `while` instruction itself. `IsBarrier`
returns true for `kWhile`, so the while stays as-is and surrounding
absorbable ops (`scan` prologue and epilogue reshapes) mega-fuse
separately on either side. Typical outcome: 1 small mega-fusion
before the while, `WhileInstruction`, 1 small mega-fusion after.

**Expected outcome.** The `WhileThunk` still drives 2,000 iterations,
but each iteration invokes a **single** `KernelThunk` (the mega-fused
body) instead of ~10 thunks. **Thunks per iteration: ~10 → 1.**

**Speedup mechanism.** 2,000 iterations × 9 fewer thunks × ~2 µs =
36 ms saved per scan call. Baseline median is tens-to-hundreds of
µs per scan (exact number depends on post-regression state); target
is ≥2× improvement. This is achievable from the dispatch reduction
alone, without any arithmetic-level changes.

**Edge case: small_while_loop_hoisting.** XLA:CPU already has a
`SmallWhileLoopHoistingPass` (`small_while_loop_hoisting_pass.cc`)
that unrolls very small while loops at compile time. If the scan
body is small enough to hoist, the hoisted form becomes a large
straight-line entry computation and Example 2 collapses into an
Example 1 shape. `MegaFusionPass` handles both cases symmetrically:
whatever the hoister produces, the pass mega-fuses. No special
casing.

## 8. File-by-File Change List

Concrete checklist. "New" means create a new file; "edit" means modify
existing.

### New files

| File                                               | Purpose                                   |
|----------------------------------------------------|-------------------------------------------|
| `xla/service/cpu/cpu_cost_model.h`                 | `OpCost`, `CpuCostModel` API (§4.1, §6)   |
| `xla/service/cpu/cpu_cost_model.cc`                | Per-opcode estimation table (§4.1 table)  |
| `xla/service/cpu/cpu_cost_model_test.cc`           | Unit tests for Estimate / IsParallelWin   |
| `xla/service/cpu/mega_fusion_pass.h`               | `MegaFusionPass`, `Options` (§4.2, §6)    |
| `xla/service/cpu/mega_fusion_pass.cc`              | Algorithm from §5                         |
| `xla/service/cpu/mega_fusion_pass_test.cc`         | Pass-level HLO→HLO tests (§9.1)           |

### Edited files

| File                                          | Change                                                          |
|-----------------------------------------------|-----------------------------------------------------------------|
| `xla/service/cpu/cpu_options.h`               | Add `kWholeProgramKflopsThreshold` constant + accessor decl.    |
| `xla/service/cpu/cpu_options.cc`              | Implement `WholeProgramKflopsThreshold` — read `DebugOptions`.  |
| `xla/xla.proto` (or wherever `DebugOptions` lives) | Add `int64 xla_cpu_whole_program_kflops_threshold` field.  |
| `xla/service/cpu/cpu_compiler.h`              | Add `std::unique_ptr<CpuCostModel> cost_model_` member.         |
| `xla/service/cpu/cpu_compiler.cc`             | Construct `cost_model_`; add `MegaFusionPass` after `FusionWrapper` block (~line 1068); include headers. |
| `xla/service/cpu/BUILD`                       | Add `cpu_cost_model` + `mega_fusion_pass` `cc_library` targets and their tests; add deps to `cpu_compiler` target. |

### Bazel build dependencies (sketch)

```starlark
# xla/service/cpu/BUILD

cc_library(
    name = "cpu_cost_model",
    srcs = ["cpu_cost_model.cc"],
    hdrs = ["cpu_cost_model.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla:shape_util",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "mega_fusion_pass",
    srcs = ["mega_fusion_pass.cc"],
    hdrs = ["mega_fusion_pass.h"],
    deps = [
        ":cpu_cost_model",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "cpu_cost_model_test",
    srcs = ["cpu_cost_model_test.cc"],
    deps = [
        ":cpu_cost_model",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
    ],
)

xla_cc_test(
    name = "mega_fusion_pass_test",
    srcs = ["mega_fusion_pass_test.cc"],
    deps = [
        ":cpu_cost_model",
        ":mega_fusion_pass",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
    ],
)

# Add to existing cpu_compiler cc_library:
#   deps += [":cpu_cost_model", ":mega_fusion_pass"]
```

### Proto field reservation checklist

1. Grep for next free field id under `DebugOptions` in `xla.proto`.
2. Add the field with a proto-doc comment matching the flag docstring.
3. Run `copybara` / proto-gen if applicable (follow existing
   `kDisableNewFusionEmitters` / `kFlattenAfterFusion` precedent).
4. Wire `WholeProgramKflopsThreshold` accessor in `cpu_options.cc` to
   read `config.debug_options().xla_cpu_whole_program_kflops_threshold()`.

### Ordering of work (suggested commits)

1. **Commit 1:** Add `CpuCostModel` + its test. No pipeline integration
   yet. Verifiable standalone.
2. **Commit 2:** Add `MegaFusionPass` + its test. Not yet wired into
   the pipeline (tests construct the pass directly and run it on
   hand-written HLO). Verifiable standalone.
3. **Commit 3:** Add the flag (proto + `cpu_options`) with default `0`
   (disabled). No behavior change.
4. **Commit 4:** Wire `MegaFusionPass` into `CpuCompiler`'s pipeline
   behind the flag. Still default-disabled.
5. **Commit 5:** Flip default to `50` (or chosen value) after
   benchmark validation.

Each commit is small and independently rolls back.

## 9. Test Plan

### 9.1 Unit tests

**`cpu_cost_model_test.cc`** — each case is ~5 lines and asserts an
exact `OpCost` for a hand-built HLO instruction:

1. Elementwise `add` on shape `[128]` → `kflops ≈ 0.128`,
   `parallelism_score == 128`.
2. `dot` of `[16,32] × [32,8]` → `kflops ≈ 2*16*32*8 / 1000 = 8.192`,
   `parallelism_score == 128`.
3. `reduce` over axis of `[1024]` → `kflops ≈ 1.024`,
   `parallelism_score == 1`.
4. `reshape` (no-op layout) → `kflops == 0`, `parallelism_score ==
   output_elems`.
5. Unknown opcode (use a custom-call or a seldom-supported opcode) →
   `kflops == 0`, `parallelism_score == INT64_MAX`,
   `IsParallelWin == true`.
6. Parallel-win threshold boundary: shape `[4095]` elementwise ⇒
   `IsParallelWin == false`; shape `[4097]` ⇒ `true`.

**`mega_fusion_pass_test.cc`** — HLO-in / HLO-out tests using
`HloHardwareIndependentTestBase`:

1. **MergesTwoAdjacentFusions.** Two kFusions, no barrier between →
   one mega-kFusion after the pass. Assert `fusion_count == 1` and
   that the merged computation contains both sets of instructions.
2. **DoesNotMergeAcrossWhile.** kFusion → while → kFusion.
   Assert `fusion_count` unchanged (2), `while` still present.
3. **DoesNotMergeAcrossCollective.** kFusion → `all-reduce` →
   kFusion. Assert unchanged.
4. **DoesNotMergeAcrossLibraryCall.** kFusion → `__onednn$matmul`
   custom-call → kFusion. Assert unchanged.
5. **SingletonRegionNotEmitted.** Single kFusion between two
   barriers. Assert no change (`min_region_size = 2`).
6. **ClosesOnKflopsOverflow.** Four kFusions each of cost 20 kflops,
   threshold = 50. Assert the pass produces two mega-fusions
   (50 and 30 kflops) not one.
7. **AbsorbsSmallDot.** kFusion → small `dot` (1 kflop, low
   parallelism because output is 1x1) → kFusion. Assert one
   mega-fusion.
8. **DoesNotAbsorbParallelizableDot.** kFusion → `dot` with
   `kflops > threshold/4` and `parallelism_score > 4096` → kFusion.
   Assert unchanged. Demonstrates §5.3 rule 5.
9. **AbsorbsLargeUnparallelizableOp.** Hand-construct an op whose
   cost exceeds `threshold/4` but whose parallelism score is tiny
   (e.g., a long reduction chain). Assert it is absorbed.
10. **WalksIntoWhileBody.** Entry has only a `while`. Body has two
    kFusions separated by nothing. Assert body gets a mega-fusion;
    entry is unchanged.
11. **PreservesComputationRoot.** Region's last member is the
    computation's root tuple's producer. Assert the new mega-fusion
    becomes the new root (or is properly wired under the root
    tuple).
12. **DisabledWhenThresholdIsZero.** `Options{0.0}`. Assert `Run`
    returns `false` and HLO is byte-identical.
13. **IdempotentOnSecondRun.** Run the pass twice; assert second
    run returns `false` (no further changes).
14. **DeterministicOutput.** Run on the same HLO twice (fresh
    modules); assert resulting HLO text is identical.
15. **MixedOpsInRegion.** Region contains mixed `kFusion` + small
    non-fusion ops (per §5.4 Path B). Assert all are absorbed into
    one mega-kFusion.

Each test loads HLO from a text literal, constructs
`CpuCostModel` + `MegaFusionPass`, runs once, and compares via
`RunAndFilecheck` or direct structural inspection.

### 9.2 Integration / benchmarks

**Integration tests** (`cpu_compiler_test.cc` or
`cpu_compiler_internals_test.cc`):

1. End-to-end compile a JAX-style HLO module matching Example A
   skeleton (5–10 small ops in a chain). With threshold = 50,
   assert the resulting thunk sequence has exactly 1 `KernelThunk`.
   With threshold = 0, assert the count matches the baseline.
2. End-to-end compile an HLO with a `while` body of small ops.
   Assert: 1 `WhileThunk`, and its body executes 1 `KernelThunk`.

**Benchmarks** — add to `xla/tools/hlo_runner_main` or the existing
CPU microbenchmark harness:

1. **bench_mass_matrix_35.** The Example A program, reduced to HLO
   (capture once via JAX HLO dump). Runs N iterations, reports
   median per-call µs. Pass if ≥ 2× faster with threshold = 50
   vs. threshold = 0.
2. **bench_scan_N2000_M3.** Example B. Same protocol, ≥ 2× bar.
3. **bench_large_matmul.** A 2048×2048 @ 2048×2048 `dot`-heavy
   program. Pass if the difference between threshold = 50 and
   threshold = 0 is within noise (± 5%). This guards the
   non-regression promise.

Benchmarks are the gate for flipping the default from `0` to `50`
(commit 5 in §8).

### 9.3 Regression guards

**Structural guards** run on every CI invocation of the pass
(via the existing `cpu_compiler_test.cc`):

1. **Thunk count non-regression.** For each HLO in a small corpus
   (`testdata/` fixtures), record baseline thunk counts and assert
   the count with `MegaFusionPass` enabled is `≤ baseline + 0`.
   (Mega-fusion can only reduce or preserve counts.)
2. **Copy-count non-regression.** Measure `kCopy` instruction count
   after copy-insertion; assert it does not grow by more than 5%
   when the pass is enabled. Flag in §5.6 to watch.
3. **Compile-time non-regression.** Time `CompileCpuExecutable` for
   a moderate-sized module; assert the pass adds < 5% compile
   time. The pass is O(instructions × avg_operand_count), so this
   should be comfortably under the bar.
4. **Determinism.** In a loop: compile the same HLO 10 times,
   assert the final LLVM IR is byte-identical. Catches
   pointer-hash iteration bugs.

**Broader guards** (run via presubmit / nightly):

5. **Existing xla:cpu test suite green** at both
   `xla_cpu_whole_program_kflops_threshold=0` and `=50`. Failing
   any test at the non-zero setting is a blocker for flipping
   the default.
6. **JAX smoketest suite** (if available in-repo) green at
   default value.
7. **Numerical regression.** For each benchmark in §9.2, compare
   outputs (with tight tolerance) between threshold=0 and
   threshold=50. Must be bit-identical for deterministic ops,
   within `atol=1e-10` for reductions (fusion may reassociate
   sums).

## 10. Rollout & Flag Semantics

## 11. Risks & Open Questions

## 12. v2 Roadmap

### 12.1 While-loop inlining

### 12.2 Parallelism-aware region splitting

### 12.3 Cost model refinement
