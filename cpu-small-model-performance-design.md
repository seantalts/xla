# Small-Model Performance on XLA:CPU: Analysis and Remediation

**Author:** seantalts · **Date:** 2026-07-08 · **Status:** Draft for review
**Code:** branch `feat/cpu-small-region-hoisting` on `seantalts/xla` (commits `c710a3a39d`, `3a1b70020e`, `aaa4c553a0`)
**Measurement record:** branch `notes/cpu-small-model-regression-findings` (findings docs plus raw agent/POC artifacts)

## TL;DR

Small scientific-computing models (ODE steppers, neuron simulators, physics engines, plasma transport codes) regressed 2x to 10x when XLA:CPU moved to the thunk runtime. The cause is not any kernel's code quality: it is granularity. These models execute tens of thousands of tiny kernels per step, and each kernel execution costs roughly 50 ns (dispatch plus call ABI) where the old runtime spent roughly 7 ns per op inside one compiled function. We propose two small passes that together restore the old runtime's shape: **region hoisting** (fold runs of small-footprint instructions, loops included, into single kernels) and **small-scatter expansion** (rewrite small scatters into loop form so they stop acting as walls between regions). Measured on the flagship bug (jax-ml/jax#26145, jaxley): 1.53 ms to 0.36 ms per step, bitwise-identical output, at or below the pre-thunk runtime's 0.372 ms. A loop re-rolling spike further shows that rolled (scan/while) programs now strictly dominate trace-time-unrolled programs on both compile time and runtime, which flips long-standing user guidance and points at follow-on work. One hardening item (a size cap on straight-line regions) is required before these passes are safe for arbitrarily large unrolled inputs.

## Background

### The workload class

The models in scope share a shape: a time-stepping or solver loop, executed hundreds to thousands of iterations per user-visible call, over arrays of tens to thousands of elements. Per-iteration math is small (4x4 solves, segment sums, elementwise updates on short vectors). Examples from the reported bugs: branched-cable neuron dynamics (jaxley), stiff ODE integration (diffrax), rigid-body kinematics (MJX), tokamak transport (TORAX), and textbook regularization terms (`jnp.diff` chains).

### What changed

The legacy XLA:CPU runtime compiled the entire entry computation into one LLVM function: while loops became native loops, and intermediate values flowed between ops inside one optimized function. The thunk runtime instead emits one kernel per instruction or fusion and executes a dependency graph of thunks. That design is right for large models (parallelism, modularity, profiling), but for the workload class above it replaces ~7 ns/op straight-line execution with ~50 ns per kernel execution (measured: ~18 ns dispatch on a calibrated synthetic slope test, plus ~30 ns of kernel ABI and prologue work at these shapes).

### The reported bugs

| issue | workload | regression | root cause (measured) | status under this proposal |
|---|---|---|---|---|
| jax#26145 | jaxley neuron sim | 4-5x per step | thunk granularity: 300-iteration while loop, ~73 thunks/iteration, scatters fragment hoistable runs | **fixed**: 1.53 -> 0.36 ms, bitwise |
| jax#33666 | diffrax stiff ODE | ~1.5-2x | LAPACK FFI calls (getrf/trsm) plus Python callback inside the Newton loop; not thunk-bound | out of scope (FFI/callback workstream) |
| jax#37465 | jnp.diff regularization | 1.7-3.2x | four scalar-emitted reduce-window kernels (~12 us of ~30 us); single-kernel codegen quality | separate fix still needed (see Non-goals) |
| jax#26021 | MJX mass matrix | 3.7x | trace-time-unrolled 35-joint loops -> ~208 DUS fusions; DUS-heavy, exceeds small-model gates | partially addressed by re-rolling direction (Sec. "Loop rolling") |
| (internal) TORAX | compile time ~3 s | n/a | call-inliner flattens ~1087 repeated call sites into a 29k-instruction module; simplification fixpoint re-sweeps it | future work: keep repeated bodies rolled (Sec. "Loop rolling") |

### Diagnosis method

Following a profile-before-coding discipline (early inferred bottlenecks changed the plan three times), we decomposed the jaxley residual with: a static thunk census of the optimized module, per-thunk trace attribution, LLVM IR audits of hot kernels, and hand-edited-HLO POCs that measure a candidate win's ceiling before any compiler work. Key numbers (Apple M3, single thread, `bench_hlo`, 100 iterations):

- Census: 300 while iterations/step x ~73 body thunks + ~144 entry thunks = ~22.4k thunk executions/step; 1.109 ms/step total; ~50 ns average per thunk execution.
- IR audit: kernel bodies are imperfect (unforwarded buffer round-trips, scalar reverse/DUS loops), yet hand-merging fusions inside kernels moved end-to-end latency ~1% (Appendix A.3). The cost is around kernels, not inside them.
- Ceiling POC: replacing scatters with passthroughs let the region hoister fold the entire program into ONE kernel: 0.238 ms/step (Appendix A.5). Scatter was the only remaining boundary op.

## Goals

1. Restore small-model runtime performance to pre-thunk levels (target: within 10% of the jax 0.4.30 runtime on the affected workloads) with bitwise-identical results.
2. Zero effect on large models: all mechanisms gated by conservative size thresholds; single flag escape hatch.
3. Keep the changes small, local to XLA:CPU, and built from existing machinery.
4. Establish the measurement harness (per-thunk attribution, seed-matched parity) so future regressions in this class are caught and diagnosed quickly.

## Non-goals

- Fixing jax#37465 (scalar reduce-window emission). It is a single-kernel codegen-quality problem, orthogonal to granularity; region folding does not change how reduce-window is emitted. The planned fix is separate: either rewrite non-overlapping reduce-window (size == stride) into reshape+reduce, or vectorize the legacy `HandleReduceWindow` path.
- Fixing jax#33666 (FFI + callback dispatch). Library calls cannot live inside a kernel; separate workstream.
- A cheaper thunk executor. At ~18 ns/dispatch the executor is already fast; the problem is count, and micro-optimizing dispatch has a ceiling of ~10% on these workloads (Appendix A.1).
- General loop re-rolling (recovering loops from already-unrolled HLO). The spike supports the direction (Appendix A.7) but detection of isomorphic subgraphs is its own project; here we only change guidance and prevention levers.

## Design overview

Three parts, in dependency order:

1. **P1: `SmallRegionHoistingPass`** (landed, `c710a3a39d`): generalizes the existing `SmallWhileLoopHoistingPass`. Partitions the entry computation and every while/conditional body into maximal runs of "region-eligible" instructions, and outlines each run into a `kCall` tagged `xla_cpu_small_call`, which the backend emits as one kernel via the existing `ComputationKernelEmitter` (whole computation in one LLVM function, control flow included). Eligibility excludes ops that need runtime services (custom-call, infeed/outfeed, sort, fft, collectives) and, before P2, scatter. Cost gate: summed bytes accessed below `max(xla_cpu_small_while_loop_byte_threshold, 64KB)`, and at least 4 members or contains control flow. Sentinel value 0 disables everything (A/B escape hatch).

2. **P2: `SmallScatterExpander`** (landed, `aaa4c553a0`): scatter is a region boundary only because the legacy emitter never implemented it (`ir_emitter.cc:1896`). XLA already owns the reference lowering: `ScatterExpander(kEliminateAllScatters)` rewrites a scatter into a while loop of gather + dynamic-update-slice, all ops the region emitter handles. We subclass it with a byte-footprint predicate (sum of operand and result tuple leaves below the same 64KB gate) and add it to the existing scatter-expansion point in the pipeline (before simplification, fusion, and copy insertion, so the expanded loops are simplified and fused normally). Small scatters stop being walls; large scatters keep the dedicated MLIR scatter kernel. On jaxley the entire step (main loop plus 54 expanded scatters) folds into one kernel: one thunk per step instead of ~22,400.

3. **P3: straight-line region size cap** (required before shipping, not yet implemented): the re-rolling spike exposed that folding a large *straight-line* chain (e.g. a 128-step trace-time-unrolled loop) into one function regresses compile time superlinearly (~N^1.8: 3x at 32 ops, 11x at 128 ops, measured 29.7 s at 512) for only ~1.4x runtime. Cause isolated by experiment: LLVM middle-end optimization of one giant basic block (opt-level 0 compiles the same function in 73 ms and keeps ~90% of the runtime win; parallel-codegen split count has zero effect). Mitigation: cap the number of members in regions that contain no control flow (loops keep basic blocks small, so loop-containing regions like jaxley's are unaffected; measured 0.34 s compile for the whole-program fold). A secondary option is a reduced LLVM opt level for oversized region functions.

### How the pieces compose

P2 exists to feed P1: expansion turns the one un-emittable op into region-eligible control flow, and P1's existing control-flow handling does the rest. Both reuse one option (`xla_cpu_small_while_loop_byte_threshold`) with one sentinel, so a single flag A/Bs the entire mechanism. P3 bounds P1's worst case on adversarially shaped (unrolled) input.

### Loop rolling: guidance change and future work

A spike (Appendix A.7) measured rolled (while/scan) versus trace-time-unrolled forms of identical computations:

- Historically (jax 0.4.30 / old runtime), unrolling was rational: identical runtime (2.6 vs 2.7 us), and unrolling only cost compile time.
- On current XLA:CPU, unrolled code is strictly worse: 4.5x slower at runtime and ~12x slower to compile than the scan form of the same program.
- On this branch, the rolled form is strictly dominant at every measured size: compile time is O(1) in trip count (~13 ms flat vs ~1 s at 512 unrolled steps) and runtime is up to 5.6x faster (the while loop folds into one kernel at ~25 ns/iteration).
- Libraries that unroll do so either out of the obsolete cost model or because of forced heterogeneity: MJX's per-joint loops dispatch on static joint type with different shapes per branch, which `lax.scan` cannot express without padding and `lax.switch`.

Consequences adopted here: (a) user-facing guidance flips to "roll your loops (scan/while); the CPU runtime penalty is gone"; (b) the most valuable follow-on workstream is *prevention* for the compile-time victims: keep repeated call sites rolled instead of letting the call inliner flatten them (TORAX: 533 call sites collapse to 11 shape signatures, ~89% of its 602 kernels), which needs no subgraph detection because the call sites still exist pre-inlining; (c) post-hoc re-rolling of already-unrolled HLO (the MJX case) is deferred: it requires isomorphic-subgraph detection, and its victims are partially served by P3 plus batching upstream fixes.

## Results

Apple M3, single thread, `bench_hlo`, 100 timed iterations, seed-matched parity. jaxley = jax#26145 `module_0272`.

| configuration | jaxley ms/step | output |
|---|---|---|
| upstream-equivalent (sentinel 0) | 1.531 | baseline |
| + P1 region hoisting | 1.09-1.12 | bitwise |
| + P2 small-scatter expansion | **0.354-0.377** | **bitwise** |
| pre-thunk runtime (jax 0.4.30, end to end) | 0.372 | reference |

Structure: optimized entry = 1 call instruction (one thunk per step); 0 scatters; 55 while loops (1 main + 54 expanded) as native control flow inside one function.

Worst-case probe for P2's predicate (a lone small scatter with no neighbors to join): 17-18 us vs 16-17 us for the dedicated scatter kernel, i.e. +1-2 us; the expanded loop itself hoists into a single kernel.

Re-rolling spike headline (synthetic stepping loop, N = 512, family 1): rolled+branch 12.7 ms compile / 27 us run; rolled+upstream 16.9 ms / 47 us; unrolled ~0.8-1.1 s compile / 152 us (either flag setting; the 64KB gate stops folding at this size).

## Testing

- Unit: 14+ tests on P1 (including bug-shaped guards modeled on #26145/#37465/#33666, token threading, control-dependency boundaries); 5 tests on P2 (small expanded, large untouched, footprint counts all operands and tuple leaves, variadic). P3 adds cap tests (straight-line region at the cap folds; above the cap splits or declines).
- Parity: every workload measurement runs seed-matched with a bitwise diff against the sentinel configuration. This caught real issues during development and is cheap; keep it in the loop.
- Non-regression: a large-scatter benchmark (predicate must not match); a large-model suite pass to confirm the byte gates keep all mechanisms inert; the compile-time matrix from the spike (unrolled N in {32,128,512}) as a compile-time regression test for P3.

## Rollout

1. Land P3 (cap) on the branch; re-run the full matrix and jaxley.
2. Upstream in two PRs: (i) P1 + P3 (region hoisting, subsumes `SmallWhileLoopHoistingPass`, which is already default-on upstream at a 1KB threshold; our change generalizes it and raises the region gate to 64KB), (ii) P2 (scatter expansion) on top. Keep the shared option and sentinel; document the escape hatch in the flag description.
3. Default posture: on, matching the existing while-hoisting precedent, with the sentinel as opt-out. If reviewers prefer staged enablement, the same option supports shipping off-by-default and flipping later; all measurements here used the final on-configuration.
4. Communicate the guidance change (roll loops on CPU) in the relevant issue threads once the PRs land in a jaxlib release.

## Monitoring and success metrics

- Success: jax#26145 closed with maintainer-reproduced numbers; no new issues attributable to the passes over one release cycle.
- Fold-rate telemetry (VLOG counters already present in the region path): regions formed, members swallowed, decline reasons. Cheap to inspect when a user reports an anomaly.
- Compile-time guardrail: the unrolled-N matrix bounds worst-case compile inflation; P3's cap keeps it within noise of upstream.

## Risks and mitigations

| risk | evidence | mitigation |
|---|---|---|
| compile-time blowup on large straight-line regions | measured ~N^1.8, 11x at 128 ops | P3 cap on control-flow-free member count; optional reduced opt level for oversized region functions (O0 retains ~90% of runtime win, 73 ms vs 2.17 s) |
| expanded scatter that never joins a region | measured +1-2 us on a lone small scatter | conservative 64KB predicate; expanded loop itself hoists; v2 option: dry-run the region-eligibility analysis before expanding |
| large-model interference (parallelism lost inside a mega kernel) | byte gates measured inert on the large-model proxies we ran | 64KB aggregate gate; explicit large-model suite run before upstreaming (open item) |
| numerics drift | none observed; all measurements bitwise | seed-matched parity in tests and in the measurement harness; expansion is XLA's reference scatter lowering |
| in-place/aliasing bugs via expanded DUS loops | none observed; expansion precedes copy insertion so standard machinery applies | pipeline position is the existing expander's; covered by upstream scatter_expander tests plus ours |

## Alternatives considered (summary)

Measured and rejected; details with numbers in Appendix A.

- Dispatch micro-optimization: per-thunk cost is already ~18 ns; ceiling ~10% (A.1).
- Tiled MLIR region emission: built and works on dense-math regions, but intra-kernel data movement is ~1% of end-to-end at these shapes; parked behind a flag (A.2).
- Intra-region fusion merging: ~1% (noise); with duplication of shared chains, 24% slower (A.3).
- In-kernel scatter emission (legacy or tiled): scatter is untileable (data-dependent writes) and the legacy implementation would be new codegen for zero benefit over expansion (A.4).
- A four-phase tiled-emitter coverage plan (multi-output, reverse, DUS, gather): superseded when measurement showed the bottleneck is granularity, not coverage (A.2).

## Open questions

1. P3 cap value: 32 vs 64 straight-line members (compile cost is 3x at 32 in the worst synthetic case; runtime sacrifice from splitting is tens of nanoseconds per boundary). Proposal: 48, revisit with the matrix.
2. Should oversized straight-line regions compile at reduced opt level instead of splitting? O0 data says most of the win survives; splitting is simpler to reason about. Proposal: split (cap) now, opt-level experiment later.
3. Upstream appetite for default-on versus staged flag flip (Rollout item 3).
4. Does TORAX's f64 working set fit the 64KB gate anywhere significant, and does the P3 cap fully protect its compile time? Verify on the real model before claiming TORAX benefits.

## References

- jax-ml/jax issues #26145, #37465, #33666, #26021
- `notes/cpu-small-model-regression-findings`: `cpu-small-model-regression-findings.md` (root-cause map), `cpu-phase05-dispatch-decomposition-findings.md` (decomposition + POC record), `phase05-artifacts/`, re-rolling spike artifacts
- `design/cpu-stage1-region-compilation`: region-emission design docs (tiled path, now appendix-status)

---

## Appendix A: POCs and alternatives that did not survive measurement

All numbers Apple M3, single thread, `bench_hlo`, seed-matched where semantics allow.

### A.1 Dispatch micro-optimization

Per-thunk dispatch measured at 15-20 ns by a slope method (synthetic K-kernel x T-iteration while loops; the slope cancels harness floor and loop overhead). An earlier estimate of 0.5-1.2 us/thunk came from a profiler artifact (nested `ThunkExecutor::Execute` frames double-count). At ~18 ns, removing ALL dispatch saves ~0.4 ms of jaxley's 1.1 ms; a faster executor cannot approach the 0.36 ms result. Rejected as primary direction.

### A.2 Tiled MLIR region emission and the four-phase coverage plan

Built (`3a1b70020e`, flag `xla_cpu_experimental_region_compilation`): routes hoisted regions through the xtile tiled emitter (SSA/stack intermediates, vectorization), with fusion-view flattening (`Defuse()` to fixpoint) and non-scalar constant lifting to call operands. Verified: a dense dot/reduce/broadcast region compiles to one tiled kernel, bitwise, ~35% faster on a synthetic region shape. On the real workloads: neutral. jaxley's regions are tuple-rooted and gather/reverse/DUS-heavy, which the tiled analysis and emitter do not cover; a four-phase plan to add that coverage (multi-output roots, reverse, dynamic-update-slice, gather) was designed, then superseded when POC-1 (A.3) showed intra-kernel data movement is worth ~1% at these shapes. Per-op findings that remain valid for future work: `SymbolicTileAnalysis` supports multi-output only in "one real root consumes the others" form; reverse propagates as a negative stride that codegen rejects; in-place DUS lowering risks an O(n) self-copy per iteration that parity tests cannot catch (latency must be measured). Status: parked behind its flag.

### A.3 Intra-region fusion merging (POC-1)

Hypothesis: region kernels lose to the old runtime because intermediates round-trip through buffer slices inside the kernel (an IR audit found ~90 of ~110 loads in the hottest region kernel re-read just-stored values; scalar reverse/DUS loops; zero vector ops in the third-hottest kernel). Test: hand-edit the optimized HLO to merge chains of fusions inside region computations so intermediates rematerialize in registers, preserving semantics. Result: duplication-free merge (the fair compiler analog) moved end-to-end latency ~1%, within noise, on both 3-region and all-region variants; a variant that duplicated shared chains ran 24% slower (1.366 ms) from elemental recompute of gather/reverse chains. All variants bitwise-identical. Conclusion: at less-than-or-equal-to-32-element shapes, in-kernel intermediate traffic is nearly free; the cost is per-kernel overhead and dispatch. This measurement killed A.2's premise.

### A.4 In-kernel scatter emission

Two routes examined. Tiled: scatter has no tile-propagation rule and cannot have one (data-dependent write indices admit no static map); confirmed by probe. Legacy: `IrEmitter::HandleScatter` is `Unimplemented`; writing new scatter codegen in the legacy emitter would duplicate what `ScatterExpander` already produces (a loop of gather/DUS), with more risk and no measured advantage: the expansion route hit the old-runtime target. Rejected in favor of P2. Note: the ceiling gap that in-kernel scatter emission could close is 0.352 -> 0.238 ms (A.5); revisit only if a workload demands it.

### A.5 Scatter passthrough ceiling (POC-2)

To measure the structural ceiling before building anything: replace all scatters with shape-correct passthroughs (semantics intentionally broken, inputs kept alive via scalar reductions) and recompile. The region hoister folded the entire entry, main loop included, into ONE kernel at default gates: 0.238 ms/step (4.6x). This localized the whole win behind scatter boundaries and motivated P2. The 0.352 vs 0.238 delta is the real scatter compute the passthroughs faked away. Related null result (POC-0): raising the byte threshold alone changed nothing; the gate was never the limiter, boundaries were.

### A.6 Wrong proxies (diagnostic history)

Recorded so future work does not repeat them: (a) the "frag" microbenchmark showed zero win from thunk collapse because its 59 thunks dispatch once, not per iteration; (b) mock_mass_matrix never folds (f64 working set exceeds the byte gate) and is DUS-bound besides; (c) a "scalar tanh smoking gun" was an artifact of an early pass routing through the legacy emitter, not the shipping path; (d) a 13x dispatch estimate came from double-counted profiler frames. Each was caught by re-measuring against the specific claim.

### A.7 Loop re-rolling spike (2026-07-08)

Question: is re-rolling unrolled loops (or preventing unrolling) a promising direction on CPU, and did library authors have a reason to unroll?

Synthetic matrix (identical math, verified bitwise across forms; loop-carried stepping body: slice -> 4x4 dot -> tanh -> dynamic-update-slice; N in {8, 32, 128, 512}; "upstream" = sentinel flags, "branch" = this proposal):

| form / metric | compile | runtime |
|---|---|---|
| rolled, branch | ~13 ms, O(1) in N | 27 us at N=512 (~25 ns/iter, single folded kernel) |
| rolled, upstream | ~17 ms | 47 us at N=512 (~70-90 ns/iter) |
| unrolled, upstream | 69 ms (N=32), 193 ms (128), 0.8-1.1 s (512) | 24/34/152 us |
| unrolled, branch | 205 ms (N=32), 2.17 s (128); gate stops folding at 512 (force-fold: 29.7 s) | 1.3-1.5x faster than upstream unrolled where folded |

Findings: (1) rolled already beats unrolled on upstream at every N on this microbenchmark, on both axes; (2) the branch widens it to strict dominance (5.6x runtime, 60-170x compile at N=512); (3) the branch's folding of large straight-line unrolled chains is a compile-time hazard (the P3 motivation): isolated to LLVM middle-end work on one giant basic block by a flag experiment (parallel-codegen split count: no effect; disable-expensive-passes: no effect; opt level 1: 1.55 s; opt level 0: 73 ms with runtime 30 vs 27 us); (4) an explicitly batched form (one [N,4]x[4,4] matmul) beats upstream forms 2-4x but ties branch-rolled at these widths, since hoisting already removed the dispatch that batching amortizes; batching should win again at larger inner dimensions.

JAX-level A/B (same computation as a Python loop vs `lax.scan`, N=128, AOT-timed): jax 0.4.30: unrolled 534 ms compile / 2.6 us run, scan 15 ms / 2.7 us (unrolling was runtime-free: the old contract). jax 0.10.1: unrolled 225 ms / 22.2 us, scan 18 ms / 4.9 us (unrolled regressed 8.5x across versions; scan barely moved). On this branch, the scan variant's HLO compiles in 21 ms and folds to a single kernel.

Why libraries unroll (source reading): MJX's per-joint loops are unrolled by construction because per-joint-type dispatch (FREE/BALL/HINGE/SLIDE with 7/4/1-wide slices) violates `lax.scan`'s homogeneity contract (`mujoco/mjx/_src/scan.py`, `smooth.py`); no unroll option exists. Where bodies are homogeneous, the same libraries roll (MJX solvers use scan/while; jaxley uses nested checkpointed scan; diffrax uses while loops). So: heterogeneity forces some unrolling, and the rest ran on an obsolete cost model.

Verdict adopted in the main doc: flip guidance, pursue prevention (keep repeated call sites rolled) as the follow-on workstream, defer post-hoc re-rolling.

## Appendix B: Implementation guide

### B.1 Code map

| commit | files | what |
|---|---|---|
| `c710a3a39d` | `xla/service/cpu/small_region_hoisting_pass.{h,cc,_test.cc}`, wiring in `cpu_compiler.cc` (~:1152) | P1: partition entry + control-flow bodies, liveness-outline runs into `xla_cpu_small_call` kCalls |
| `3a1b70020e` | `xla/backends/cpu/codegen/emitters/region_kernel_emitter.{h,cc}`, `tiled_fusion_emitter.cc`, `thunk_emitter.cc` routing | tiled region emission, flag-gated (parked; A.2) |
| `aaa4c553a0` | `xla/service/cpu/small_scatter_expander.{h,cc,_test.cc}`, wiring in `cpu_compiler.cc` (~:931) | P2: byte-predicated `ScatterExpander(kEliminateAllScatters)` subclass |
| (todo) | `small_region_hoisting_pass.cc` | P3: straight-line member-count cap |

### B.2 Pass mechanics

`SmallRegionHoistingPass(byte_threshold, min_region_size=4, exclude_nonscalar_constants=false)`:
- Walks each computation's instruction sequence in topological order, splitting at boundary instructions (`InstructionIsUnavailable`: custom-call, infeed/outfeed, scatter, sort, fft, partition/replica-id, custom fusions, collectives, recursively through called computations).
- For each maximal run: liveness analysis decides parameters (values used from outside) and results (values used outside the run, or root); multi-output runs get a tuple root plus get-tuple-element consumers. Regions whose members have control dependencies crossing the boundary are skipped.
- Cost gate: summed `bytes_accessed` of members < threshold AND (member count >= min_region_size OR the run contains while/conditional). P3 adds: if the run contains no control flow, member count must also be <= cap.
- Control-flow bodies are enqueued for partitioning after their parent, so a swallowed while's body is not re-partitioned.
- The outlined call carries frontend attribute `xla_cpu_small_call`; `thunk_emitter.cc:663` routes it to `ComputationKernelEmitter` (or `RegionKernelEmitter` under the experimental flag, with clean fallback).

`SmallScatterExpander(small_buffer_access_size)`:
- `InstructionMatchesPattern` = base match (all scatters, `kEliminateAllScatters` mode) AND static byte footprint (all operand shapes + all result tuple leaves) < threshold; dynamic shapes decline.
- Pipeline position: `cpu_compiler.cc` scatter sandwich, in the fusion-emitter branch, immediately before `post_scatter_expansion_simplification`, so expanded loops are simplified, fused, and copy-inserted by the standard pipeline.

### B.3 Flags and options

- `xla_cpu_small_while_loop_byte_threshold` (backend extra option): shared threshold. Unset -> pass defaults (region gate `max(option, 1<<16)`); `0` -> disables P1 and P2 entirely (the A/B and escape hatch used for every baseline in this doc).
- `xla_cpu_experimental_region_compilation` (backend extra option): tiled region emission (parked; off by default).
- User workaround available before these passes ship: `XLA_FLAGS=--xla_cpu_use_fusion_emitters=false` forces unconditional scatter expansion and recovers jax#26145 today, at the cost of disabling MLIR fusion emitters globally (not recommended for large models).

### B.4 Reproducing the measurements

`bench_hlo` (dev tool, `xla/tools/bench_hlo.cc`, kept out of the fix commits) compiles an HLO text file once and reports per-iteration wall time; `--profile` prints per-thunk attribution; `--print_result --seed=42` gives deterministic parity checks.

```
# jaxley end to end (branch vs upstream-equivalent)
bench_hlo --hlo_file=jaxley-dump/module_0272.jit_run.before_optimizations.txt \
  --iters=100 --warmup=10 --print_result --seed=42
XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0" \
  bench_hlo --hlo_file=... (same)
# structure check: --xla_dump_to, then count entry instructions / " scatter(" / " while(" in *after_optimizations*.txt
```

Raw artifacts (agent findings, POC scripts, the re-rolling generator and matrix JSON) live on `notes/cpu-small-model-regression-findings`.

### B.5 Upstream PR slicing

1. PR 1: P1 + P3, replacing `SmallWhileLoopHoistingPass` (which it strictly generalizes; port its tests, keep its option). Include the compile-time matrix as a regression test.
2. PR 2: P2 on top (small file, ~60 lines + tests), with the jaxley numbers in the description.
3. Not proposed upstream now: the tiled region emitter (parked), bench_hlo (dev tool).

### B.6 Remaining work checklist

- [ ] P3 cap (design above; re-run matrix + jaxley after)
- [ ] Large-model suite verification run (gates inert, no compile-time movement)
- [ ] jax#37465 reduce-window fix (cheap-alternative spike first: reshape+reduce rewrite for size == stride)
- [ ] TORAX prevention lever scoping (keep repeated call sites rolled; measure compile-time recovery against the 533-sites-to-11-shapes census)
- [ ] Re-measure jax#26021 (MJX) once P3 lands; evaluate whether capped folding plus guidance suffices or post-hoc re-rolling is worth designing
