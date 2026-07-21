# XLA:CPU kernel emission modernization: roadmap and design

**Author:** seantalts · **Date:** 2026-07-16 (updated 2026-07-21) · **Status:** Draft for review

**Audience:** XLA:CPU contributors and reviewers of the codegen roadmap. The document assumes familiarity with HLO (XLA's high-level IR), MLIR, and the general structure of the XLA:CPU backend. The glossary (section 8) defines the terms this document uses in a specific sense.

**Scope:** the path from current XLA head to the end state: no legacy emitters, no fixed fusion blockers, kernel boundaries chosen by a cost model, and the small-model performance class fixed along the way. Out of scope: GPU emission (except where the infrastructure is shared), performance inside the libraries themselves, and JAX-side API changes.

Numbers labeled "prototype" come from a working implementation of the M0 passes (Apple M3, single thread, bitwise-identical outputs unless noted).

## 1. Summary

XLA:CPU has finished one modernization (the thunk runtime) and is midway through several others (MLIR fusion emitters, the xtile tiled emitter, library integration). The unfinished state has a measurable cost. Small scientific models regressed 2x to 10x against the pre-thunk runtime because they execute tens of thousands of tiny kernels per step, and each kernel pays ~50 ns of fixed overhead. Reported instances include [jax-ml/jax#26145](https://github.com/jax-ml/jax/issues/26145), [jax-ml/jax#37465](https://github.com/jax-ml/jax/issues/37465), [jax-ml/jax#33666](https://github.com/jax-ml/jax/issues/33666), [jax-ml/jax#26021](https://github.com/jax-ml/jax/issues/26021), and [jax-ml/jax discussions#24501](https://github.com/jax-ml/jax/discussions/24501). Meanwhile the legacy pre-thunk emitter cannot be deleted, because it still backs hoisted-call kernels and several op fallbacks. Every improvement built on it deepens the dependency.

Measurement keeps this story honest in both directions. The census in section 5.1 found its sharpest single number on the modern path, not the legacy one: a 553x recompute pathology in the current MLIR loop emitter, with zero amplification involved. Every deletion and default flip in this plan is therefore gated on measurement, not on which path is newer.

This document proposes the following end state:

1. **One kernel construct.** A fusion and a hoisted call become the same thing: a set of instructions emitted as one function. A cost model decides, for each producer-consumer edge, whether the value is recomputed at each use, kept in a tile-sized stack allocation, or written to a scratch buffer. Materialization therefore decouples from fusion membership: whether an intermediate is written out is a per-edge decision, not a consequence of where the kernel boundary fell. Multi-output fusion becomes one point on this spectrum rather than a special mechanism.
2. **No fixed blockers.** Every op that today cannot join a kernel either gains a modern emission path or an HLO expansion. Two classes remain permanent boundaries: ops that suspend execution (infeed/outfeed, collectives), and opaque runtime services (FFI custom calls, host callbacks, sort, fft). Folding works around the second class rather than through it unless the optional M5 lands.
3. **Libraries stay thunks.** The thunk runtime made library calls first-class execution units on purpose. For any op worth sending to a library, ~50 ns of thunk overhead is noise against microseconds of work, so that design stays. What changes: library-preferred ops stop being hard fusion blockers. Inside an amplified small kernel, the same dot or reduction emits natively through xtile, at sizes where native code is competitive anyway. Native codegen is the default; a library is chosen only where it is clearly better (large dots and convolutions), at thunk granularity.
4. **xtile as the only backend.** A whole-computation driver (control flow, multi-output, materialization) and op-coverage completion land on xtile. `IrEmitter`, `IrEmitter2`, and `ComputationKernelEmitter` are then deleted.
5. **Cost-modeled boundaries.** A static, machine-parameterized model replaces the fixed thresholds accumulated along the way: byte budgets, size gates, protection lists, and module caps.

Headline prototype numbers: the neuron-simulation workload of [jax#26145](https://github.com/jax-ml/jax/issues/26145) goes from 1.53 ms to 0.36 ms per step, at parity with the pre-thunk runtime; the million-iteration scan of [discussions#24501](https://github.com/jax-ml/jax/discussions/24501) improves 6-18x.

## 2. Background: the cost model of the thunk runtime

The thunk runtime executes one kernel (or copy, or library call) per HLO instruction or fusion, through a dependency-graph executor. Each kernel execution costs about 50 ns at small sizes: ~18 ns of dispatch plus ~30 ns of kernel-call ABI (application binary interface) work (prototype measurement, calibrated slope method). The pre-thunk runtime compiled the entire program into one LLVM function. It spent roughly 7 ns per op and ran while loops as native loops.

For large models, the thunk design is right. Per-kernel granularity buys intra-op parallelism, inter-kernel overlap, profiling, and library integration, and 50 ns vanishes against millisecond kernels. For a small model whose step is a loop over arrays of tens of elements, the fixed overhead dominates: [jax#26145](https://github.com/jax-ml/jax/issues/26145) executes ~22,400 thunks per 1.1 ms step. (That figure is the issue repro at head defaults. The census capture of the same workload in section 5.1, measured with hoisting disabled, executes ~54,000; the two configurations are not directly comparable.)

The decisive variable is **amplification**: how many times per user-visible call a piece of code runs. Code inside a while loop pays its fixed overheads once per iteration. A 50 ns saving is therefore worth 50 ms inside a million-iteration scan, and nothing measurable in straight-line entry code. Every design in this document that folds work into bigger kernels spends compile time and parallelism options to eliminate amplified fixed overhead. The gating question is always whether the amplification pays for what is given up.

## 3. State of modernization at head

The following table summarizes each modernization project, its state at current head, and the gap this roadmap must close.

| project | status at head | remaining gap |
|---|---|---|
| Thunk runtime | complete, default | none; it is the substrate |
| MLIR fusion emitters (`--xla_cpu_use_fusion_emitters`, default on) | standalone loop fusions emit through the modern path, including a dedicated MLIR scatter emitter | reduce-window still falls back to the legacy scalar path (the whole cost of [jax#37465](https://github.com/jax-ml/jax/issues/37465), ~12 us of a ~30 us program); the `=false` mode keeps the entire legacy pipeline alive; per-use recompute of shared producers is pathological on deep unrolled chains (553x vs the legacy emitter on an MJX kinematics module, 5.1) |
| xtile tiled emitter | analysis and driver exist (`SymbolicTileAnalysis`, `EmitTiledComputation`); dense-math subgraphs compile behind an experimental flag; tile-sized intermediates already become stack allocations via bufferization | op coverage (no reduce-window, gather, reverse, dynamic-update-slice; reverse propagates as a negative stride that emission rejects); multi-output only when one root consumes the others; tiling propagation for dot-containing graphs is experimental; no control flow; single-threaded |
| Library integration (oneDNN, Eigen, YNNPACK) | thunk-level only: dots, convolutions, and claimed reductions run as library thunks | claiming gates are fixed constants (e.g. a 4096-element minimum that [jax#37465](https://github.com/jax-ml/jax/issues/37465)'s reduction misses by 64 elements); a claimed op inside a loop blocks folding the loop |
| Autotuning | backend-agnostic framework (`xla/backends/autotuner/`) with production GPU backends (cuDNN/cuBLAS/Triton/native selection, Triton tilings); a CPU implementation exists (`xla/backends/cpu/autotuner/`) | not connected to any decision this doc adds; see 4.5 for where it fits |
| `SmallWhileLoopHoistingPass` | upstream, default on: outlines a small while (<1 KB accessed) into a call compiled as one kernel | whiles only, 1 KB only; M0 generalizes it to arbitrary small runs |
| Legacy `IrEmitter` / `IrEmitter2` / `ComputationKernelEmitter` | the backend for hoisted-call kernels and assorted fallbacks | the thing this roadmap deletes |

## 4. Design

### 4.1 One kernel construct: fusion = hoisted call, materialization per edge

Today the two kernel-forming mechanisms sit at opposite extremes of one axis:

- A **fusion** recomputes per use: elemental emission, no intermediate buffers.
- A **hoisted call** materializes everything: each member writes its buffer slice inside one function.

Both extremes have measured costs. Recompute of shared chains cost +24% in a hand-edit experiment, and up to 553x where deep unrolled chains turn the shared-producer DAG (directed acyclic graph) into a tree (the MJX kinematics module of 5.1). Full materialization is nearly free at 32-element sizes (prototype) but leaves medium-size intermediates round-tripping memory.

The right answer is per edge. For each producer-consumer edge, the cost model chooses one of three placements:

1. Recompute the producer at each use (cheap producers).
2. Keep the value in a tile-sized stack allocation (the xtile bufferization already does this at <=4 KB).
3. Spill the value to a scratch buffer (larger intermediates).

A multi-output fusion is then just an edge someone chose to materialize at the kernel boundary. One construct, one emitter, one cost model choosing per edge. The design requires a kernel scratch-buffer arrangement: hoisted calls get one because buffer assignment recurses into the called computation; pure fusions need an equivalent.

The ordering between per-edge decisions and buffer allocation resolves at HLO time, not emission time. The boundary former (M0's pass, later the cost model) records materialization choices as edge annotations before buffer assignment runs. Any kernel with a scratch-materialized edge takes the call form, which buffer assignment already knows how to recurse into. Emission consumes annotations and never invents buffer demand. The pure-fusion syntax therefore needs no new scratch mechanism, and per-edge activation (M3) does not wait on the ABI freedom that arrives at M4.

### 4.2 Library and runtime calls: thunks by design, native emission inside kernels

The thunk runtime made library calls first-class execution units deliberately. A oneDNN dot or a claimed convolution runs as its own thunk with profiling, intra-op parallelism, and a clean buffer contract. For any op worth sending to a library, the ~50 ns of thunk overhead is noise against microseconds of work. Nothing here changes that, and no mid-kernel library-call machinery is needed for performance.

The real conflict is narrower than "libraries block fusion". It bites only when a library-claimed or otherwise unemittable op sits inside an amplified loop, where the op acts as a hard kernel boundary and prevents folding the loop around it. For the library-preferred op classes (dot, reduction, convolution), the resolution is to emit the op natively instead of calling the library from inside the kernel:

- Inside a foldable small kernel, the operands are small by construction.
- xtile's per-op emitters already cover dot and reduction.
- At those sizes, native code is competitive with or better than a library call plus its dispatch.

The cost model (4.5) makes this a per-site decision. At thunk granularity, library placement wins where the calibrated table says so. Inside a kernel, the op emits natively, or the kernel boundary falls at the op. "Library ops block fusion" stops being an architectural fact and becomes a sizing decision.

What remains is the opaque residue: runtime calls with no HLO decomposition and no emitter. Examples: FFI (foreign function interface) custom calls, such as the LAPACK factorizations inside [jax#33666](https://github.com/jax-ml/jax/issues/33666)'s Newton loop; host callbacks, such as the progress-bar variant of [discussions#24501](https://github.com/jax-ml/jax/discussions/24501); and sort and fft. Two mitigations need no new ABI:

- **Segmentation (M0).** Folding the runs between opaque ops leaves a loop body that contains one paying a handful of dispatches per iteration instead of one per instruction.
- **Small-size HLO expansion.** Specific residents can retire. JAX already carries an HLO-level LU decomposition used where LAPACK is unavailable, and sorting networks are the same move for small sorts, at exactly the sizes that fold.

If a workload still demands folding through an opaque op, the escape hatch is a runtime-context kernel ABI. One pointer to a per-execution context (run options, intra-op threadpool, allocator, FFI execution context, profiler hooks) is added to the kernel calling convention. Call sites resolve at emission time into prebuilt call frames, invoked through a stable C trampoline. The default is a sequential context; a library that manages its own parallelism receives the pool from the context rather than nesting. The pre-thunk emitter's `allow_runtime_calls` mechanism proves feasibility: it is still present in the code, disabled for hoisted-call kernels precisely because the ABI lacks a context. This ABI is scheduled last (M5) and built only on demand, because everything else in this roadmap is simpler and pays first. In-kernel copies lower to `llvm.memcpy` regardless, so folding never trades away the memcpy fast path.

### 4.3 The xtile computation driver

The driver is a schedule-walking emitter that takes an outlined computation and produces one MLIR function:

- Members emit in order through the existing xtile per-op emitters.
- `scf.while` and `scf.if` provide control flow, with loop-carried values as memrefs.
- Tuple roots map to multiple result buffers.
- Intermediates are placed per 4.1.

The driver reuses the existing tile analysis, the experimental whole-computation emitter's fusion-view flattening and constant-lifting mechanics, and the bufferization pipeline.

The parity bar for replacing the legacy backend is dual, because a single bitwise bar would structurally reject the wins this roadmap sells:

- Where emission preserves evaluation order and association, outputs must be bitwise-identical.
- Where vectorization legitimately reassociates (reductions above all), the bar is tolerance-based, with explicit per-op sign-off recorded in the harness.

Both halves require equal-or-better latency and compile time on the full regression corpus.

Compile time is a first-class bar because one large function is the known failure mode. LLVM middle-end cost grows ~N^1.8 in straight-line body size (prototype: 2.17 s for a 128-step unrolled chain, against 73 ms at opt-level 0 with ~90% of the runtime win retained). The driver therefore owns emission-size discipline and opt-level tiering.

The tiling infrastructure underneath is shared property, not a CPU fork. `SymbolicTileAnalysis` and its data structures live in `xla/codegen/tiling/` and serve both the CPU tiled emitter and GPU's Triton compiler (`xla/backends/gpu/codegen/triton/`). GPU is actively expanding the same analysis: in-tree TODOs track removing the single-real-root restriction on multi-output fusions (b/372454662, b/390569102), which is exactly the relaxation item 2 of 4.4 needs, and broadening op support. Analysis-side coverage work in this roadmap should therefore land in the shared code, where GPU co-maintains it and may close some gaps first. Only the emission side (the driver and the CPU per-op lowerings) is CPU-owned.

### 4.4 New emitters and capabilities (the build list)

1. **Computation driver** (4.3): control flow, multi-output, scheduling, materialization. The largest single piece.
2. **Multi-output tiling:** `SymbolicTileAnalysis` currently accepts multiple roots only when one "real root" consumes the others; hoisted-call shapes have independent roots. Under whole-shape single tiles there is no cross-root consistency problem, so the relaxation is contained.
3. **Tiled reduce-window:** window dimensions in the tiling space, the windowed affine map in propagation, a masked reduce emission; vectorization then comes free from the existing reduce lowering. Permanently fixes [jax#37465](https://github.com/jax-ml/jax/issues/37465). (An independent interim fix, an HLO rewrite of non-overlapping reduce-window to reshape+reduce, can ship earlier.)
4. **Tiled gather, reverse, and dynamic-update-slice:** gather emits per output row as a dynamic-slice of the operand at a runtime index, scoped to observed forms; reverse normalizes the negative stride the analysis already produces; dynamic-update-slice is the write-side dual of the supported dynamic-slice, with the in-place case gated on latency rather than parity, because a naive lowering inserts a full-array self-copy per loop iteration that parity checks cannot see.
5. **In-kernel sequential scatter (optional):** a loop over index rows under whole-shape tiles; no tiling relation needed. Built only if a workload demands the last measured gap (0.352 vs 0.238 ms on [jax#26145](https://github.com/jax-ml/jax/issues/26145)); small-scatter expansion already covers the class.
6. **Mid-kernel runtime-call emission** (4.2): optional, M5, demand-driven; only for the opaque residue.
7. **Parallel tiles:** tile loops mapped onto the existing kernel workgroup dimensions, so one large kernel uses the intra-op pool. This lets kernel sizes grow without losing parallelism, replacing per-op partitioning for fused work.
8. **Memcpy-aware copy emission.**

### 4.5 The cost model

One cost function serves three clients: kernel-boundary formation, per-edge materialization, and library placement. It replaces today's fixed constants.

The benefit of a fold is saved thunk executions x ~50 ns x amplification (loop trip counts are usually known statically from `known_trip_count`), plus locality gains for tile-materialized edges. The costs are:

- **Recompute:** per shared producer, uses x producer cost; exact from the graph.
- **Inter-kernel overlap forfeited:** subgraph total work minus critical path, capped by pool width.
- **Intra-op parallelism forfeited:** would the members partition if they ran standalone?
- **Register pressure:** the max live cut of the per-element expression vs the physical register file; decides recompute-vs-materialize per edge.
- **Operand streams:** stream count vs the hardware prefetcher budget.
- **Working set:** vs cache capacities.
- **Library-vs-codegen deltas:** from a small calibrated table.
- **Code size.**
- **Compile-time budget:** member and basic-block caps as non-negotiable rails.

Every term is computable to order of magnitude from the graph plus a machine description (dispatch constant, ABI constant, register file, cache sizes, stream budget, pool width). The model's job is not precision; its job is to avoid catastrophic corners. Every regression found during prototyping is a corner a coarse model catches:

- Folding copy-only runs: +633% and +1321% on two workloads (the kernel ABI cost exceeded the dispatch saved).
- Serializing independent once-per-call fusions: -13% on a 1B-parameter transformer.
- Blocking loop hoisting to protect fusions for newer emitters: +550% on a high-trip-count scan.
- Gating on whole-module instruction count: +37% on a 3,011-instruction model, because the gate was calibrated on benchmark slices smaller than production modules.

Two standing lessons are encoded as constraints. First, gates must be properties of the folded run itself, never of the whole module: benchmark HLOs are often partial-model slices, so module-level thresholds do not transfer to production. Second, the model must be pinned by a continuously-run anchor suite plus production fold telemetry, or it rots.

Autotuning complements the static model rather than replacing it. XLA already has a backend-agnostic autotuner (`xla/backends/autotuner/`): on GPU it selects per instruction among cuDNN, cuBLAS, Triton, and native emission, and searches Triton tile configurations; a CPU implementation exists (`xla/backends/cpu/autotuner/`). It fits in two places:

1. **Calibration.** Autotuning-style microbenchmarks produce the machine description once per platform (the dispatch and ABI constants, the library-vs-codegen table), instead of hand-maintained values.
2. **Per-kernel search at M3.** For the few decisions where the static model's error bars are wide and the kernel is hot (tile sizes for large kernels, contested library-vs-native placements), the autotuner searches a space the static model prunes and defaults.

Autotuning cannot do the small-model job itself. That class has thousands of kernels of tens of nanoseconds each, so per-kernel search costs more compile time than it can ever recover, and the folding decision changes the kernel set being tuned underneath the search. There the static model decides, and autotuning only supplies its constants.

### 4.6 What remains a boundary

Infeed/outfeed and collectives suspend execution and stay at thunk granularity. Opaque runtime services (FFI custom calls, host callbacks, sort, fft) stay boundaries by default: segmentation folds around them, and only the optional M5 folds through them. Large library-claimed ops stay thunks because the cost model places them there, not because they are blockers. Everything else is eventually admissible.

## 5. Milestones

Each milestone below states what it changes, how to implement it, and what gets measurably faster when it lands. M1 can start immediately against the hoisted calls head already produces. M5 is optional and may never be needed.

### M0: generalize hoisting from whiles to runs

Replace `SmallWhileLoopHoistingPass` with a pass that outlines any small run of adjacent instructions, including whole whiles and while bodies, into calls compiled as single kernels. Expand small scatters at HLO so they stop breaking runs. A prototype of exactly this milestone produced the measurements quoted throughout. The milestone ships as two PRs: hoisting replaces the while pass and ports its tests; scatter expansion stacks on it.

*Implementation:*

- **Partitioning.** The pass walks each computation's schedule and partitions it into maximal runs at boundary ops. "Boundary" is a recursive check: the op, or anything in its nested computations, needs runtime services or has no kernel emission.
- **Budgeting.** A run whose summed `HloCostAnalysis` bytes-accessed fits the byte budget outlines whole. An over-budget run is greedily segmented at oversized members and at budget overflow.
- **Segment survival.** A segment survives only if it is amplified or contains control flow. Once-per-call straight-line segments are discarded: folding them serialized inter-thunk overlap, measured at -13% on a transformer.
- **Amplification.** Computed by worklist: entry is unamplified, while bodies and conditions are amplified, and conditional branches inherit from their parent.
- **Outlining.** Liveness-based, into a kCall carrying a backend-config marker. The called computation is emitted as one kernel by the same backend that serves the while pass today.
- **Scatter expansion.** A `ScatterExpander` subclass whose match predicate adds a byte-footprint bound over the operands and result. It runs before fusion so the expanded scalar graph folds into the surrounding runs, and it is enabled if and only if hoisting is (one shared threshold).
- **Pipeline position.** After parallel task assignment in `cpu_compiler.cc`; skipped under the fast-compile preset.

*Faster:* small stepping models with rolled loops improve 3-4x ([jax#26145](https://github.com/jax-ml/jax/issues/26145): 1.53 to 0.36 ms, bitwise). Loop-dominated scans improve up to 18x ([discussions#24501](https://github.com/jax-ml/jax/discussions/24501)). Any small `lax.scan`/`while_loop` body drops from ~70-90 ns to ~25 ns of overhead per iteration. User guidance flips to "roll your loops on CPU": on jax 0.4.30 a rolled and an unrolled loop ran at the same speed; at head the unrolled form is 4.5x slower and ~12x slower to compile; with M0 the rolled form wins on both axes at every size. Cost: +14% compile time on transformer-shaped models where per-layer bookkeeping folds (bounded, and excluded under the fast-compile preset); large-tensor models are unaffected by construction.

### M1: the xtile computation driver

Build the driver of 4.3: control flow, multi-output, per-edge materialization, memcpy-aware copies. Then perform two separately gated backend flips, in order of risk:

1. Hoisted calls switch to the driver, behind the existing experimental flag and then by default at the parity bar. This retires `ComputationKernelEmitter`.
2. Standalone loop-path fusions switch from the MLIR loop emitter to xtile. This is the larger flip (526 of 526 census emissions, 5.1 finding 2), and the 5.1 finding-1 recompute class is its named acceptance test.

*Implementation:*

- **Driver.** A new emitter beside the tiled fusion emitter. It takes the outlined computation plus buffer assignment and walks the schedule.
- **Lowering.** Each member lowers through the existing xtile per-op emitters. Intra-kernel edges become tile tensors that bufferize to stack allocations, the default middle point of 4.1. This is also the structural fix for the recompute pathology of 5.1 finding 1: a shared producer computes once per tile instead of once per use.
- **Control flow.** While lowers to `scf.while` with loop-carried values as memrefs; conditional lowers to `scf.if`; bodies emit recursively through the same walker.
- **ABI mapping.** A tuple root maps to multiple result buffers in the kernel ABI. Nonscalar constants lift to operands. The experimental whole-computation emitter already has both mechanics. Copies lower to `llvm.memcpy`.
- **Compile-time tiering.** The opt level keys on straight-line member count (opt-level 0 measured 73 ms vs 2.17 s on a 128-step chain, keeping ~90% of the runtime win).
- **Slicing.** Each stage lands alone: driver skeleton on straight-line bodies, then control flow, then multi-output, then per-edge materialization. The hoisted-call flip needs only the first two stages. The standalone-fusion flip additionally needs the coverage set the census attributes (broadcast, slice, reduce, concatenate, nonscalar constants; 5.1 finding 2). That set is mostly emission-side lowering, because the shared analysis already handles broadcast, reduce, and dot.
- **Gating.** The parity harness compiles every corpus program under both backends and applies the dual bar of 4.3: bitwise where association is preserved, signed-off tolerance where vectorization reassociates. It then compares latency and compile time. Each flag flip gates on winning both, and no gate is inferred from the other flip's result.

*Faster:* direct wins on existing folds are modest, because the legacy backend is already adequate at 32-element sizes (a measured ~35% on a synthetic dense subgraph from vectorization and stack intermediates; more on medium shapes). The qualitative win: loops containing small dots and reductions fold for the first time, because the op emits natively in-kernel instead of splitting the loop at a library thunk (4.2). Beyond that, every M2 op lands once instead of twice, the protected-fusion conflict disappears (hoisted calls emit through the same modern emitters), and the legacy dependency is broken. Compile-time tiering (opt level by kernel size) lands here.

### M2: op coverage on the modern path

Add tiled reduce-window, gather, reverse, and dynamic-update-slice; optionally add in-kernel scatter. Each op ships with a regression test shaped like its originating bug and a latency gate. Order per the census (5.1): dynamic-update-slice, then gather, then reverse; reduce-window is issue-driven.

*Implementation:*

- **Reduce-window.** Add window dimensions to the tiling space and the windowed affine map to propagation, then emit a masked reduce. Vectorization comes free from the existing reduce lowering, under the tolerance half of the 4.3 bar, since it reassociates. The independent interim fix, an HLO rewrite of non-overlapping windows to reshape+reduce, can ship before any of this. It only helps [jax#37465](https://github.com/jax-ml/jax/issues/37465) if that program's windows are non-overlapping, which must be verified against its HLO: the repro is jnp.diff plus sum, and lowering introduces the reduce-windows, so overlap is not knowable from the Python.
- **Gather.** Emit per output row as a dynamic-slice of the operand at a runtime index, under whole-shape tiles. Scope to the canonical forms the simplifier produces.
- **Reverse.** Normalize the negative stride the analysis already propagates. Emission-side only.
- **Dynamic-update-slice.** The write-side dual of the supported dynamic-slice. Gate the in-place case on measured latency, because a naive lowering inserts a full self-copy per iteration that parity checks cannot see.
- **Scatter (optional).** A sequential loop over index rows under whole-shape tiles; no tiling relation needed.
- **Where the code lands.** Each op lands with its originating issue's HLO as a regression test plus a latency gate against the thunk path. Analysis-side pieces land in the shared `xla/codegen/tiling/` code (4.3).

*Faster:* [jax#37465](https://github.com/jax-ml/jax/issues/37465)-class programs improve ~1.5x (the four scalar reduce-window kernels, ~12 us of ~30 us, become vectorized). [jax#26021](https://github.com/jax-ml/jax/issues/26021)-class dynamic-update-slice-heavy models improve (its 3.7x regression is DUS-dominated; the DUS work item plus M3 sizing target it). Scatter-heavy loops gain the last ~1.5x if in-kernel scatter ships (0.35 to ~0.24 ms on [jax#26145](https://github.com/jax-ml/jax/issues/26145)).

### M3: cost-modeled boundaries and large kernels

The quantitative model replaces the M0 gates. Fusion sizing and call hoisting unify under it, per-edge materialization activates, and parallel tiles let big kernels keep the thread pool. The module gate and the protection lists are deleted.

*Implementation:*

- **Machine description.** A struct of dispatch and ABI constants, register file, cache sizes, prefetch stream budget, and pool width, produced by a calibration binary built on the autotuner framework (4.5) and checked in per platform class.
- **One cost library, three clients.** The boundary former (which replaces the M0 byte budget, member cap, and minimum size with model queries), per-edge materialization inside the M1 driver, and library-vs-native placement.
- **Quantitative amplification.** `known_trip_count` is plumbed through, so amplification becomes a number rather than a boolean.
- **Parallel tiles.** The driver's tile loops map onto the existing kernel workgroup dimensions. The model owns the per-kernel choice between internal parallelism and executor overlap.
- **Hard rails.** The compile-time budget and the register cap stay outside the model as non-negotiable bounds.
- **Shadow mode first.** The model ships log-only against the M0 heuristics, recording every disagreement on the anchor suite and in production telemetry. The three clients then flip one at a time (boundary former, then per-edge materialization, then library placement). An M0 gate is deleted only after its replacement client has survived a release.
- **Anchor suite.** The issue repros of section 6 plus a transformer anchor run continuously with bitwise, latency, and compile-time assertions.

*Faster:* medium models: per-layer chains get properly sized kernels instead of threshold-shaped ones. Trace-time-unrolled code (the [jax#26021](https://github.com/jax-ml/jax/issues/26021) shape) folds safely under the compile rails, measured 1.3-1.5x on synthetic unrolled chains. Large-tensor fusions grow without serializing; today's byte gates forbid exactly the folds that would pay there. This is also where compile-time regressions become structurally impossible rather than gated: the budget is a model input.

### M4: delete the legacy emitters

Deletion is inventory-driven. The hoisted-call backend died at M1 and the reduce-window fallback at M2. The `use_fusion_emitters=false` mode retires with a deprecation window. Then `IrEmitter`, `IrEmitter2`, and `ComputationKernelEmitter` are removed, and the pipeline's dual paths (for example, the unconditional scatter-expansion branch) collapse.

*Implementation:*

- **Measure the inventory; never assume it.** Compile a representative corpus with per-instruction logging of which emitter handled each instruction. Rank the legacy consumers by hit count, weighted by amplification and by known performance gaps.
- **Work items.** Each surviving consumer gets a work item in M1 or M2.
- **Hard-error protocol.** When a consumer's count reaches zero across the corpus, its fallback becomes a hard error for one release, so stragglers surface as actionable reports rather than silent legacy use. Then the code deletes.
- **Which corpus authorizes what.** The corpus that authorizes a hard-error flip is the production-derived one described at the end of 5.1, not the local census. The local census (5.1) orders the work.

*Faster:* nothing directly. The payoff is maintenance, single-path testing, and the ability to change the kernel ABI (M3's scratch buffers, M5's optional context) without double implementation.

### M5 (optional, demand-driven): mid-kernel runtime calls

Build the runtime-context ABI of 4.2: context parameter, trampoline registry, FFI call frames. Sort and fft then leave the boundary list. This milestone is built only if a workload appears that folding-around (M0 segmentation) and HLO expansion cannot satisfy. The reentrancy machinery does not exist until then.

*Implementation:*

- **ABI.** One pointer parameter appended to the kernel calling convention, carrying run options, the intra-op pool, the allocator, the FFI execution context, and profiler hooks.
- **Call frames.** Call sites resolve at emission time into call frames stored with the executable, invoked through a stable C trampoline keyed by target symbol.
- **Threading.** Sequential-context default with explicit pool handoff.
- **Landing order, if built.** The ABI change first (it is also the last ABI change M4 makes cheap), then sort/fft as the low-risk in-tree targets, then FFI frames, then host callbacks.

*Faster:* loops containing FFI custom calls fold completely ([jax#33666](https://github.com/jax-ml/jax/issues/33666), a 1.5-2x regression: the Newton loop folds with LAPACK called in-kernel). Loops containing host callbacks stop splitting at every iteration (the progress-bar variant of [discussions#24501](https://github.com/jax-ml/jax/discussions/24501), reported at 13 to 25 s). Loops containing sort or fft fold for the first time.

### 5.1 Legacy-path census on the local corpus

This census measures which emitters actually run on real programs at head, to rank the M4 deletion order and the M2 op order. Method: five locally captured programs were recompiled from pre-optimization HLO at head-equivalent defaults, with while hoisting disabled so every op routes individually. Per-emitter counts come from emission logging, amplification from the optimized module structure, and runtimes plus per-thunk attribution from a benchmark harness (Apple M3, single thread).

The corpus has three known limits, stated here rather than silently:

- The torax module optimizes to a single copy thunk and is excluded.
- No transformer-class module is retained locally. IrEmitter2's largest remaining production surface (dot-rooted and output fusions) therefore has zero instances below, and its deletion rank is provisional.
- The diffrax module compiles its HLO but cannot execute in this harness (no LAPACK FFI handlers registered), so it contributes counts, not runtimes.

The following table shows the thunk-level composition of each optimized module. "Amplified" means inside a while body.

| program | fusions (amplified) | dominant fusion roots | other thunk-level ops | wall/call |
|---|---|---|---|---|
| jaxley `jit_run` | 274 (168, trip count 300) | scatter 54, gather 52, divide 32, DUS 24, slice 19, reverse 8 | 34 copies, 3 DUS, 2 concatenates, 1 while | 1.58 ms |
| diffrax `jit_solve` | 73 (66) | select 17, add 8, reduce 4, DUS 3 | 32 copies, 6 custom-calls (4 amplified LAPACK), 4 dots (amplified f32[1,8] GEMV), 4 whiles, 2 conditionals | n/a |
| mjx `jit_kinematics` | 176 (0; fully unrolled) | reduce 58, concatenate 40, add 19, pad 8, scatter 8, gather 5 | 73 copies, 5 concatenates | 529 ms (see below) |
| mm `jit_mock_mass_matrix` | 255 (0) | DUS 208, bitcast 34, pad 4 | 40 dots: 35 f64[4,4] Eigen thunks, one f64[35,35] claimed by YNNPACK, one f64[3,3] naive legacy LLVM | 0.072 ms |

Which emitters actually fired: the MLIR loop/concatenate/DUS emitters carried every fusion emission on the loop path (526 across the three fully compiled modules, with kernel dedup saving 57/48/1 re-emissions). Scatter-rooted fusions went to the MLIR scatter emitter. IrEmitter2 fired 3 times (jaxley in-place DUS), ConcatenateKernelEmitter 7 times (standalone concatenates), and DotKernelEmitter 5 times (diffrax's 4 amplified GEMVs, mm's 3x3). ElementalKernelEmitter fired zero times. ComputationKernelEmitter fired zero times with hoisting off; the hoisting pass is its only producer.

The findings below are ordered by how much they change the plan.

1. **The worst offender is on the modern path, not the legacy path.** mjx kinematics runs 529 ms at head defaults and 0.957 ms with fusions forced to the legacy elemental emitter via `xla_cpu_disable_new_fusion_emitters`: 553x. The hot kernels are 567-571-instruction fusion bodies that produce f32[1,3] outputs in 208 ms and 136 ms per execution. Per-kernel times decay geometrically along the kinematic chain, consistent with per-use recompute turning the shared-producer DAG into a tree (exponential in chain depth); the legacy emitter memoizes elemental values per index. jaxley (1.58 vs 1.52 ms) and mm (0.072 ms both ways) show no gap. A first reduction attempt localizes the defect: a synthetic pure-elementwise chain with two uses per value does not reproduce (LLVM common-subexpression elimination covers it; 0.015 vs 0.020 ms), while the hot bodies share values through slices (single values used five times via slice-of-shared-subexpression, 269 multiplies for a 3-element output), so the recompute survives CSE through differing index expressions. Three consequences follow. First, this is an immediate work item tracked outside this roadmap's milestones: an upstream filing plus a targeted per-index memoization in the loop emitter (which is what the legacy emitter does), plausibly days of work against M1's quarters, and the highest return per line in this document. Second, the M1 driver's stack-allocation default for intra-kernel edges is the structural fix, and this class is the named acceptance test for the standalone-fusion flip. Third, deleting the legacy fusion path (M4) is gated on this class measuring dead, not on coverage parity alone.
2. **xtile currently emits nothing, and the reason is the op allow-list, not the analysis.** 0 of 526 loop-path fusion emissions were claimed by the tiled emitter, with tiling propagation on or off. Attribution over the 715 loop-path fusion instances: the CPU emitter admits only elementwise ops, bitcast/reshape/transpose/iota, and scalar constants, while the corpus fusions contain slice (279 instances), broadcast (267), dynamic-update-slice (254), concatenate (117), nonscalar constants (95), gather (90), reduce (75), and reverse (52). The shared `SymbolicTileAnalysis` already handles broadcast, reduce, and dot; only the allow-list and per-op lowerings gate them. The rest is emission-side work. About 50 fusions contain only whitelisted ops and still fell back, an unattributed analysis-side residue worth a look before the M1 flip. Consequence: the standalone-fusion flip's prerequisite coverage (broadcast, slice, reduce, concatenate, nonscalar constants) is distinct from M2's issue-driven list and sits inside M1's second flip, which is why that flip is the larger one.
3. **ElementalKernelEmitter is already unreachable on real programs.** The fusion wrapper turns every surviving standalone elemental op into a single-op loop fusion. The standalone emitter therefore lives only through its fallback conditions (strided copy, non-Eigen convolution, layout-mismatched concatenate, non-in-place DUS), and none of them fired. It is the first hard-error candidate in the M4 protocol.
4. **Amplification is concentrated where M0 aims.** jaxley executes about 54,000 kernel dispatches per 1.58 ms call (180 amplified thunks times 300 trips), all sub-microsecond kernels. An arithmetic note: 54,000 times the ~50 ns constant of section 2 exceeds the wall time because dispatches overlap across the executor's thread pool; the constant is a sequential calibration, so it prices the CPU work of dispatch and is not a divisor of wall time here. diffrax has 90% of its kernels amplified, including its legacy-emitted GEMV dots and its LAPACK calls (the M0 fold-around case, M5's anchor). mjx and mm have zero amplification; their costs are kernel quality (finding 1) and library placement (finding 5) respectively, which M0 does not touch.
5. **Small-dot placement, measured.** mm makes 40 standalone dot calls per 72 us step. 35 of them are f64[4,4] Eigen library thunks, the exact shape 4.2 proposes to emit natively once dots sit inside folded kernels.

The census ranks the deletion and coverage work as follows. For M4 deletion order:

1. ElementalKernelEmitter's standalone path and ConcatenateKernelEmitter go to hard-error first (7 unamplified corpus hits between them).
2. DotKernelEmitter waits for M1 native dot emission; its hits are amplified.
3. IrEmitter2's in-place DUS retires with the M2 DUS item.
4. IrEmitter2's fusion path goes last: it is currently the safety net for finding 1, and the missing transformer-class measurement lands on it.

For M2 op order: dynamic-update-slice first (235 rooted fusions corpus-wide, 27 amplified), gather second (57, 17 amplified), reverse third (8, all amplified). Reduce-window stays issue-driven ([jax#37465](https://github.com/jax-ml/jax/issues/37465) is the anchor; zero corpus instances).

One constraint the document imposes on itself applies here. Section 4.5 encodes that thresholds calibrated on partial slices do not transfer, and this census is four programs of one workload class. It is sufficient to order work items; it is not sufficient to flip a fallback to hard-error. Hard-error flips and final deletions are gated on a production-derived corpus (fleet compilation telemetry and the internal benchmark suites, plus modules donated on the tracked issues) confirming zero hits, with the transformer-class gap closed first for anything touching IrEmitter2's fusion path.

## 6. Issue map

The following table maps each reported issue to the mechanism behind it and the milestone that closes it.

| issue | mechanism | closed by |
|---|---|---|
| [jax-ml/jax#26145](https://github.com/jax-ml/jax/issues/26145) (jaxley neuron sim, 4-5x) | thunk count in a 300-iteration loop; scatters fragment foldable runs | M0 (prototype measured: bitwise, at pre-thunk parity); M2 optional scatter adds ~1.5x |
| [jax-ml/jax discussions#24501](https://github.com/jax-ml/jax/discussions/24501) (scan + progress bar) | million-iteration scan next to hoisted invariants; callback splits the loop | M0 (6-18x, no-callback); callback variant mitigated by M0 segmentation, fully M5 |
| [jax-ml/jax#33666](https://github.com/jax-ml/jax/issues/33666) (diffrax stiff, 1.5-2x) | LAPACK FFI + Python callback inside the Newton loop | M0 partially (runs between the opaque ops fold); fully M5, or an HLO LU expansion at small sizes |
| [jax-ml/jax#37465](https://github.com/jax-ml/jax/issues/37465) (jnp.diff, 1.7-3.2x) | scalar legacy reduce-window emission | interim HLO rewrite if its windows are non-overlapping (unverified against the issue's HLO); permanently M2 |
| [jax-ml/jax#26021](https://github.com/jax-ml/jax/issues/26021) (MJX mass matrix, 3.7x) | trace-time-unrolled joint loops, DUS-heavy, medium tensors | M2 (DUS) + M3 (sizing); library-side batching is a separate upstream conversation |

## 7. Risks

- **M1 parity.** Risk: the legacy emitter is battle-tested on control flow, and the xtile driver replays that maturation. Mitigation: the parity harness and regression corpus already exist; keep the legacy fallback flag one full release after parity.
- **Modern-path inversions.** Risk: the census found a 553x case where the current modern emitter loses to the legacy one (5.1 finding 1), so "modern" cannot be assumed faster per class. Mitigation: latency gates are per program and measured, never inferred from which path is newer; the legacy fusion path deletes last.
- **Cost-model overfit.** Risk: anchors are finite, and the model can drift from production reality. Mitigation: hard rails (compile budget, register cap) are non-negotiable regardless of model output; gates read properties of the folded run only; production fold telemetry is the drift alarm.
- **Coverage scope creep (M2).** Risk: op coverage grows without bound. Mitigation: op work is issue-driven with per-op measurement gates. An earlier coverage plan on this project was cancelled by measurement after one week of scoping, and its findings are reused here rather than relitigated.
- **Parallel tiles vs executor parallelism (M3).** Risk: one kernel becomes both internally parallel and concurrently overlapped, oversubscribing the pool. Mitigation: the cost model owns the either/or choice per kernel.
- **M5 reentrancy (if built).** Risk: library calls from kernels can nest pools or deadlock. Mitigation: sequential-context default, explicit pool handoff via the context, a nested-scan stress test, and the [jax#33666](https://github.com/jax-ml/jax/issues/33666) repro as the anchor. The demand gate on M5 is itself the primary mitigation.

## 8. Glossary

- **Thunk:** the runtime's unit of execution, roughly one kernel launch per instruction or fusion. A thunk execution costs ~50 ns at small sizes (~18 ns dispatch + ~30 ns kernel-call ABI).
- **Amplification:** how many times per user-visible call a piece of code executes. While bodies (and anything nested under them) execute trip-count times; entry code executes once. Fixed overheads only matter where amplified, which is the central discriminator for every folding decision here.
- **Run:** a maximal sequence of schedule-adjacent instructions with no boundary op inside it; the unit the hoisting pass considers.
- **Hoisted call:** a run outlined into a call and compiled as one kernel. `SmallWhileLoopHoistingPass` creates these for small whiles at head; M0 generalizes them to arbitrary runs; after M1 they emit through xtile; after M3 "hoisted call" and "fusion" are the same construct.
- **Fold:** absorb an instruction into a kernel so it stops being a separately dispatched thunk.
- **Materialize / rematerialize:** whether a producer's value is written somewhere (buffer or stack) and reloaded, or recomputed at each consumer use. Elemental fusion rematerializes; hoisted calls materialize to buffer slices; the end state chooses per edge.
- **Boundary:** an op that cannot join a kernel. Today: runtime-service ops, library-claimed ops, and coverage gaps. End state: execution-suspending ops, plus opaque runtime services unless the optional M5 lands.
- **xtile:** the modern MLIR tiled-emission stack (`SymbolicTileAnalysis`, tile propagation, `EmitTiledComputation`, bufferization to stack allocations).
- **Fusion emitters:** the MLIR emitters for standalone fusions (`--xla_cpu_use_fusion_emitters`, default on at head), distinct from the xtile tiled path and from the legacy elemental emitters.
- **Partial-model slice:** a benchmark HLO cut from a larger program. Module-level properties measured on slices do not transfer to production modules; per-run properties do.
