# SmallRegionHoistingPass: single-kernel compilation for small program regions

**Author:** seantalts · **Date:** 2026-07-09 · **Status:** Implemented, in review for upstreaming
**Code:** [`c710a3a3`](https://github.com/seantalts/xla/commit/c710a3a39d) on [`feat/cpu-small-region-hoisting`](https://github.com/seantalts/xla/tree/feat/cpu-small-region-hoisting)
**Parent doc:** [XLA:CPU performance on small scientific computing models](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-small-model-performance-design.md)

## Objective

Restore near-legacy-runtime performance for small models on XLA:CPU by compiling runs of adjacent small instructions, loops included, into single kernels. Measured on [jax#26145](https://github.com/jax-ml/jax/issues/26145): 1.53 ms to 1.10 ms per step from this pass alone, 0.36 ms with the companion scatter expander, bitwise-identical output.

## Background

The thunk runtime executes one kernel per instruction or fusion. Each kernel execution costs about 50 ns: ~18 ns of dispatch plus ~30 ns of call ABI and prologue work at small array sizes. A small model with a stepping loop (hundreds of iterations, dozens of ops per iteration) executes tens of thousands of kernels per step; jax#26145 executes ~22,400 for 1.1 ms of wall time, so overhead dominates compute. The legacy runtime compiled the whole program into one LLVM function and spent ~7 ns per op. This class of model (neuron simulators, ODE steppers, physics engines) regressed 2x to 10x in the migration.

XLA:CPU already contains a narrow version of the fix: `SmallWhileLoopHoistingPass` outlines a small while loop into a call tagged `xla_cpu_small_call`, and `ComputationKernelEmitter` emits the callee as one kernel through the legacy `IrEmitter`, which handles control flow natively. It only fires on individual while loops, so it misses straight-line runs, loop bodies, and everything between.

## Design

`SmallRegionHoistingPass` generalizes the while-loop pass from "a small while loop" to "any small region".

**Partitioning.** The pass walks the entry computation in schedule order and splits it into maximal runs of region-eligible instructions. Eligibility excludes ops that need runtime services, checked recursively through called computations: custom-call, infeed/outfeed, scatter, sort, fft, partition/replica-id, custom fusions, and collectives. After partitioning a computation, the pass enqueues the bodies of any while/conditional it did *not* swallow and partitions those the same way; a loop that was swallowed rides along inside its region and is not re-partitioned.

**Outlining.** For each accepted run, liveness analysis determines the interface: values defined outside and used inside become call parameters; values defined inside and used outside (or the root) become results, wrapped in a tuple when there is more than one. The run is outlined into a new computation called via a `kCall` tagged `xla_cpu_small_call`. Runs whose members have control dependencies crossing the region boundary are skipped rather than risking a reorder.

**Cost gates.** A run is hoisted only if (a) the summed `bytes_accessed` of its members is under `max(xla_cpu_small_while_loop_byte_threshold, 64KB)`, and (b) it has at least 4 members or contains control flow. Setting the option to 0 disables the pass entirely; this sentinel is the A/B and escape hatch used for every baseline below. Gate (a) keeps large models untouched: their fusions exceed the byte budget and continue to run thunk-granular, preserving inter-kernel parallelism.

**Straight-line size cap (required addition).** Folding a large straight-line chain into one function inflates compile time superlinearly (~N^1.8: 3x at 32 ops, 11x at 128, measured on trace-time-unrolled loops). Flag ablation attributes the cost to LLVM middle-end optimization of one giant basic block; codegen split count has no effect, and opt-level 0 compiles the same function in 73 ms instead of 2.17 s while keeping ~90% of the runtime win. Regions containing loops are unaffected because their basic blocks stay small (the jax#26145 whole-program fold compiles in 0.34 s). Fix: cap member count for runs with no control-flow member (proposed cap: 48). Splitting an oversized run costs tens of nanoseconds per extra boundary, so the runtime sacrifice is negligible.

**Emission.** Unchanged from the while-loop pass: `thunk_emitter.cc` routes tagged calls to `ComputationKernelEmitter`, one kernel per region. The pass is emission-agnostic HLO surgery; an experimental MLIR region emitter can replace the backend later without touching it. It runs at the end of the HLO pipeline, after copy insertion, so it sees final program structure.

## Results

Apple M3, single thread, seed-matched bitwise parity against the sentinel configuration.

| workload | before | after (this pass) | note |
|---|---|---|---|
| jax#26145 step | 1.531 ms | 1.09-1.12 ms | 0.354-0.377 ms with the scatter expander; legacy-runtime reference is 0.372 ms |
| rolled loop microbenchmark, per iteration | ~70-90 ns | ~25 ns | loop folds into one kernel; compile time flat in trip count |

## Testing

14+ unit tests: partitioning and outlining, boundary ops stay outside regions, token threading (infeed/outfeed in loop bodies), crossing control dependencies block hoisting, multi-output tuples, and regression tests shaped like the three reported issues. Every benchmark run doubles as a correctness check via `--print_result --seed` bitwise diffs. The unrolled-N compile-time matrix from the [re-rolling spike](https://github.com/seantalts/xla/tree/notes/cpu-small-model-regression-findings/reroll-spike) becomes a regression test for the size cap.

## Rollout

Upstream as a replacement for `SmallWhileLoopHoistingPass`, which it strictly generalizes: port its tests, keep its option and default-on posture, raise the region gate to 64KB (the 1KB default remains the option floor for anyone who set it). Ships together with the straight-line cap. Escape hatch documented on the option.

## Risks

- Compile time on unrolled input: covered by the cap above; regression-tested.
- Large-model interference: byte gate keeps the pass inert; verified on the large-model proxies we ran, with a full suite run scheduled before the upstream PR.
- Codegen quality inside regions: the legacy emitter emits scalar loops for some ops. Measured irrelevant at small sizes (hand-merging fusions inside regions moved end-to-end time ~1%), and the region boundary is where a future MLIR emitter slots in.

## Alternatives considered

- Faster thunk dispatch: already ~18 ns; ceiling ~10% on the affected models.
- Bigger fusions instead of regions: fusion cannot cross control flow or multi-output boundaries, which is exactly where the cost lives; and measured intra-kernel gains at these sizes are ~1%.
- Tiled/MLIR region emission as the v1 backend: built behind a flag, measured neutral at small sizes; kept as the long-term backend, not a v1 dependency.

## Open questions

1. Cap value: 48 proposed; the compile-cost curve says 32 is already 3x in the worst case, but the runtime cost of splitting is negligible either way.
2. Cap vs reduced opt level for oversized straight-line regions: data supports both; the cap is simpler to reason about, opt-level tiering can come later with the MLIR emitter.
