# SmallScatterExpander: design and implementation guide

**Author:** seantalts · **Date:** 2026-07-13 · **Status:** Implemented on branch; this doc is the handoff for porting, hardening, and upstreaming
**Code:** [`aaa4c553`](https://github.com/seantalts/xla/commit/aaa4c553a0) on [`feat/cpu-small-region-hoisting`](https://github.com/seantalts/xla/tree/feat/cpu-small-region-hoisting)
**Companion:** [SmallRegionHoistingPass design](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-small-region-hoisting-pass-design.md) · **Parent:** [small-model performance doc](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-small-model-performance-design.md)

## Does this stand alone?

No, as a performance feature; yes, as a code artifact. Understand this before touching anything else:

- Expanding a scatter replaces one dedicated kernel with a while loop of gather and dynamic-update-slice. If region hoisting does not then fold that loop into a single kernel, the loop executes as a `WhileThunk` dispatching several thunks per index row: strictly worse than the kernel it replaced. The pass exists only to convert an op the region emitter cannot compile into control flow it can.
- The invariant to preserve in any port or refactor: **this pass must never be enabled where `SmallRegionHoistingPass` is not.** Today that is structural, not conventional: both passes read the same option (`xla_cpu_small_while_loop_byte_threshold`) with the same sentinel (0 disables both), and any preset gating applied to the hoisting pass must apply here too.
- As an artifact it is self-contained: one ~60-line pass, five unit tests, one wiring block. It reviews and lands as its own PR, stacked on the hoisting PR.

## Background

Scatter is a region boundary because the legacy kernel emitter never implemented it: `IrEmitter::HandleScatter` returns `Unimplemented` ([`ir_emitter.cc:1896`](https://github.com/seantalts/xla/blob/feat/cpu-small-region-hoisting/xla/service/cpu/ir_emitter.cc)). On [jax#26145](https://github.com/jax-ml/jax/issues/26145) that boundary chops each while-body iteration into ~18 regions plus 31 scatter thunks; a placeholder POC showed that with scatters out of the way the hoister folds the entire program into one kernel at 0.238 ms/step (vs 1.10 with hoisting alone). XLA already owns the lowering we need: `ScatterExpander(kEliminateAllScatters)` is the reference rewrite of scatter into a while loop of gather and dynamic-update-slice, used as the fallback whenever a backend lacks scatter support. Forcing it on this workload gave 0.352 ms with bitwise-identical results before any new code was written. The only missing piece was selectivity: large scatters should keep the dedicated MLIR scatter fusion kernel.

## Design

Subclass, do not fork:

```cpp
class SmallScatterExpander final : public ScatterExpander {
 public:
  explicit SmallScatterExpander(int64_t small_buffer_access_size)
      : ScatterExpander(kEliminateAllScatters),
        small_buffer_access_size_(small_buffer_access_size) {}
  absl::string_view name() const override { return "small-scatter-expander"; }
 protected:
  bool InstructionMatchesPattern(HloInstruction* inst) override;
 private:
  int64_t small_buffer_access_size_;
};
```

`InstructionMatchesPattern` = the base class match (all scatters, since the mode is `kEliminateAllScatters`) AND a byte-footprint predicate: sum `ShapeUtil::ByteSizeOf` over the array leaves of every operand shape and of the result shape (tuple leaves cover variadic scatters), and require the total to be under the threshold. Any dynamic-shaped leaf declines. All expansion mechanics (loop construction, combiner application order, in-place semantics) are inherited from the base class, which is what makes the numerics argument short: this is XLA's reference scatter lowering, applied selectively.

**Threshold:** `max(xla_cpu_small_while_loop_byte_threshold, 1 << 16)`, sentinel 0 disables. Deliberately identical to the region-hoisting gate: a scatter small enough to expand is by construction small enough that its expanded loop and neighbors can fold.

**Pipeline position** ([`cpu_compiler.cc` ~line 931](https://github.com/seantalts/xla/blob/feat/cpu-small-region-hoisting/xla/service/cpu/cpu_compiler.cc)): the existing scatter-expansion sandwich, in the fusion-emitter branch, complementary to the `!use_fusion_emitters` block that already runs the unconditional expander, and immediately before `post_scatter_expansion_simplification`. Position is the second thing to preserve in any port: expansion must precede simplification (the expanded loops get cleaned up), fusion (expanded components fuse normally), and copy insertion (loop-carried aliasing needs standard copy semantics). Running it later breaks all three for free.

```cpp
if (use_fusion_emitters && kFusionEmitterScatterEnabled) {
  ASSIGN_OR_RETURN(int64_t small_scatter_byte_threshold,
                   xla::cpu::options::SmallWhileLoopByteThreshold(module->config()));
  if (small_scatter_byte_threshold != 0) {
    pipeline.AddPass<SmallScatterExpander>(
        std::max<int64_t>(small_scatter_byte_threshold, int64_t{1} << 16));
  }
}
```

## Implementation guide

Files (all under `xla/service/cpu/`): `small_scatter_expander.h`, `small_scatter_expander.cc`, `small_scatter_expander_test.cc`; a `cc_library` + `xla_cc_test` in `BUILD` (deps: `//xla/service:scatter_expander`, `//xla:shape_util`, `//xla/hlo/ir:hlo`); the wiring block above plus the header include and the `:small_scatter_expander` dep on `cpu_compiler_pure`.

Write the tests first and confirm they fail to build before implementing (the original was built RED-GREEN; keep it that way in a port):

1. `SmallScatterExpanded`: f32[16] operand, s32[4,1] indices, f32[4] updates, add combiner. Pass returns true; no `kScatter` remains; a `kWhile` exists.
2. `LargeScatterUnchanged`: same structure, f32[200000] operand (800 KB). Returns false; scatter untouched.
3. `FootprintCountsIndicesAndUpdates`: operand small, indices+updates large enough to cross the threshold. Not expanded.
4. `VariadicSmallScatterExpanded`: two-operand variadic scatter under the gate expands.
5. `VariadicScatterFootprintCountsAllOperands`: each operand pair is under the gate (40 KB) but the sum (80 KB, operands plus tuple result) is over. Not expanded.

HLO-text gotchas that will waste an hour if unknown: instruction names must not be a type prefix plus digit (`s1`, `f2`) or a lexer keyword (`inf`); scatter combiners are module-scope computations.

**Verification protocol** (all measured values from the current branch; Apple M3, single thread; `bench_hlo` is the dev tool at `xla/tools/bench_hlo.cc`, kept out of fix commits):

```
# jaxley end to end: expect ~0.35-0.38 ms/step and bitwise-identical output vs sentinel
bench_hlo --hlo_file=jaxley-dump/module_0272.jit_run.before_optimizations.txt \
  --iters=100 --warmup=10 --print_result --seed=42
XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0" \
  bench_hlo ... (same; expect ~1.5 ms)
# structure via --xla_dump_to: after-opt entry = 1 call instruction, 0 " scatter(", 55 " while("
```

| check | expected |
|---|---|
| jaxley step, both passes on | 0.354-0.377 ms, bitwise vs sentinel |
| jaxley structure | entry = 1 call; 0 scatters; 55 whiles (1 main + 54 expanded) |
| lone small scatter, nothing to fold with | +1-2 us vs dedicated kernel (the expanded loop folds alone) |
| Gemma3 1B (`gemma3_1b_flax_call`, in-tree registry) | all 22 tiny attention scatters expand; runtime wash; bitwise ([record](https://github.com/seantalts/xla/blob/notes/cpu-small-model-regression-findings/gemma3-1b-large-model-verification.md)) |
| large-scatter benchmark | predicate must not match; latency unchanged |

## Risks

- **Expanded but not hoisted.** The coupling risk described up top. Bounded today by the shared gate and measured at +1-2 us in the worst constructed case; the systematic fix, if it is ever needed, is a v2 predicate that dry-runs the hoisting pass's region-eligibility analysis ("would this scatter fragment an otherwise-hoistable region") before expanding. Not built; do not build it without a workload that demands it.
- **Numerics.** Inherited reference lowering, sequential combiner order preserved. Measured bitwise on jaxley and Gemma3 1B. Keep the seed-matched parity diff in every measurement run.
- **Compile time.** Expansion adds a small while per scatter; the measured branch-level compile deltas (2.2x on the jaxley whole-program fold, +13% on Gemma3 1B) include both passes, and the preset gating on the hoisting pass (skip under `CPU_OPT_PRESET_FAST_COMPILE`) must cover this pass identically.

## Alternatives considered

- **In-kernel scatter emission** (implement `HandleScatter` in the legacy emitter, or a tiled-emitter case): scatter is not tileable (data-dependent write indices), and legacy codegen would reproduce by hand what the expander already generates. Deferred to the M3 region-emitter work, where it would close the measured 0.352 vs 0.238 ms gap.
- **Unconditional expansion** via `--xla_cpu_use_fusion_emitters=false`: works today (it is how the win was first measured) but disables the MLIR fusion emitters globally; usable as a user workaround, not shippable.
- **Leaving scatters as boundaries** (hoisting alone): the shipped Stage-0 state; 1.10 ms vs 0.36 ms on jax#26145. Insufficient.
- **Smarter predicates** (expand only inside while bodies; dry-run region eligibility): rejected for v1 in favor of the simplest gate that matches the hoisting pass; the worst-case measurement justifies the simplicity.

## Upstreaming

Stack on the region-hoisting PR; do not land first or separately enabled. The PR is ~60 lines plus tests; put the jaxley before/after numbers and the parity method in the description. Port `ScatterExpander`'s existing tests untouched; this pass adds behavior only through the predicate, so upstream's expansion-correctness coverage carries over.
