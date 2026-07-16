# XLA:CPU kernel emission modernization: roadmap and design

**Author:** seantalts · **Date:** 2026-07-16 · **Status:** Draft for review
**Scope:** everything between current head and the end state: no legacy emitters, no fixed fusion blockers, libraries callable from inside kernels, kernel boundaries chosen by a cost model, and the small-model performance class fixed along the way. Written as a delta from current XLA head; prototype measurements come from a development branch ([`feat/cpu-small-region-hoisting`](https://github.com/seantalts/xla/tree/feat/cpu-small-region-hoisting)) and are labeled as such. This doc is self-contained; a glossary is at the end.

## 1. Summary

XLA:CPU finished one modernization (the thunk runtime) and is midway through several others (MLIR fusion emitters, the xtile tiled emitter, library integration). The unfinished state has a measurable cost. Small scientific models regressed 2x to 10x against the pre-thunk runtime because they execute tens of thousands of tiny kernels per step at ~50 ns of fixed overhead each; reported instances include [jax-ml/jax#26145](https://github.com/jax-ml/jax/issues/26145), [jax-ml/jax#37465](https://github.com/jax-ml/jax/issues/37465), [jax-ml/jax#33666](https://github.com/jax-ml/jax/issues/33666), [jax-ml/jax#26021](https://github.com/jax-ml/jax/issues/26021), and [jax-ml/jax discussions#24501](https://github.com/jax-ml/jax/discussions/24501). Meanwhile the legacy pre-thunk emitter is still load-critical: it is the backend for hoisted-region kernels and several op fallbacks, so it cannot be deleted, and every improvement built on it deepens the dependency.

The end state this doc proposes:

1. **One kernel construct.** A fusion and a compiled region are the same thing: a set of instructions emitted as one function, where a cost model decides per producer-consumer edge whether a value is recomputed at each use, kept in a tile-sized stack allocation, or materialized to a scratch buffer. Multi-output fusion becomes one point on that spectrum rather than a special mechanism.
2. **No fixed blockers.** Every op that today cannot join a kernel either gains a modern emission path, an HLO expansion, or a mid-kernel library/runtime call. The only permanent boundaries are ops that suspend execution (infeed/outfeed, collectives).
3. **Libraries callable mid-kernel.** A runtime-context kernel ABI lets emitted code call oneDNN, Eigen, YNNPACK, LAPACK/FFI handlers, and host callbacks from inside a kernel, as the pre-thunk emitter did with Eigen for years. Design principle: native codegen is the default and a library is chosen only where it is clearly better (large dots and convolutions); this work enables placement decisions, it does not route more work to libraries.
4. **xtile as the only backend**, after a region driver (control flow, multi-output, materialization) and op-coverage completion land on it. `IrEmitter`, `IrEmitter2`, and `ComputationKernelEmitter` are then deleted.
5. **Cost-modeled boundaries.** A static, machine-parameterized model replaces the fixed thresholds accumulated along the way (byte budgets, size gates, protection lists, module caps).

Headline prototype numbers (Apple M3, single thread, bitwise-identical outputs, dev branch): the neuron-simulation workload of [jax#26145](https://github.com/jax-ml/jax/issues/26145) goes from 1.53 ms to 0.36 ms per step, at parity with the pre-thunk runtime; the million-iteration scan of [discussions#24501](https://github.com/jax-ml/jax/discussions/24501) improves 6-18x.

## 2. Background: the cost model of the thunk runtime

The thunk runtime executes one kernel (or copy, or library call) per HLO instruction or fusion, through a dependency-graph executor. Each kernel execution costs about 50 ns at small sizes: ~18 ns of dispatch plus ~30 ns of kernel-call ABI work (measured by a calibrated slope method on the dev branch). The pre-thunk runtime compiled the entire program into one LLVM function and spent roughly 7 ns per op, with while loops as native loops.

For large models the thunk design is right: per-kernel granularity buys intra-op parallelism, inter-kernel overlap, profiling, and library integration, and 50 ns vanishes against millisecond kernels. For a small model whose step is a loop over arrays of tens of elements, the fixed overhead dominates: [jax#26145](https://github.com/jax-ml/jax/issues/26145) executes ~22,400 thunks per 1.1 ms step. The decisive variable is **amplification**: code inside a while loop pays its fixed overheads once per iteration, so a 50 ns saving is worth 50 ms inside a million-iteration scan and nothing measurable in straight-line entry code. Any design in this doc that folds work into bigger kernels is, at bottom, spending compile time and parallelism options to eliminate amplified fixed overhead, and the gating question is always whether the amplification pays for what is given up.

## 3. State of modernization at head

| project | status at head | remaining gap |
|---|---|---|
| Thunk runtime | complete, default | none; it is the substrate |
| MLIR fusion emitters (`--xla_cpu_use_fusion_emitters`, default on) | standalone loop fusions emit through the modern path, including a dedicated MLIR scatter emitter | reduce-window still falls back to the legacy scalar path (the whole cost of [jax#37465](https://github.com/jax-ml/jax/issues/37465), ~12 us of a ~30 us program); the `=false` mode keeps the entire legacy pipeline alive |
| xtile tiled emitter | analysis and driver exist (`SymbolicTileAnalysis`, `EmitTiledComputation`); dense-math regions compile behind an experimental flag; tile-sized intermediates already become stack allocations via bufferization | op coverage (no reduce-window, gather, reverse, dynamic-update-slice; reverse propagates as a negative stride that emission rejects); multi-output only when one root consumes the others; tiling propagation for dot-containing graphs is experimental; no control flow; single-threaded |
| Library integration (oneDNN, Eigen, YNNPACK) | thunk-level only: dots, convolutions, and claimed reductions run as library thunks | unreachable from inside any kernel; claiming gates are fixed constants (e.g. a 4096-element minimum that [jax#37465](https://github.com/jax-ml/jax/issues/37465)'s reduction misses by 64 elements) |
| `SmallWhileLoopHoistingPass` | upstream, default on: outlines a small while (<1 KB accessed) into a call compiled as one kernel | whiles only, 1 KB only; the seed of region compilation |
| Legacy `IrEmitter` / `IrEmitter2` / `ComputationKernelEmitter` | the backend for hoisted-region kernels and assorted fallbacks | the thing this roadmap deletes |
| Region compilation (dev branch, not yet upstream) | region hoisting generalizes the while pass to any small run including loop bodies; small-scatter expansion removes the most common boundary; budget segmentation with amplification gating; measured results below | upstreaming (M0); still emits through the legacy backend |

## 4. Design

### 4.1 One kernel construct: fusion = region, materialization per edge

Today "fusion" means recompute-per-use (elemental emission, no intermediate buffers) and "region" (dev branch) means materialize-everything (each member writes its buffer slice inside one function). Both extremes are measured: recompute of shared chains cost +24% on a hand-edit experiment; full materialization is nearly free at 32-element sizes but leaves medium-size intermediates round-tripping memory. The right answer is per-edge: recompute cheap producers, keep tile-sized values in stack allocations (the xtile bufferization already does this at <=4 KB), and spill larger intermediates to scratch buffers. A multi-output fusion is then just "an edge someone chose to materialize at the kernel boundary." One construct, one emitter, one cost model choosing per edge. This requires a kernel scratch-buffer arrangement (regions get it from buffer assignment recursing into the called computation; pure fusions need an equivalent).

### 4.2 The runtime-context kernel ABI and mid-kernel calls

The kernel calling convention gains one pointer to a per-execution context: run options, intra-op threadpool, allocator, FFI execution context, profiler hooks. Library and FFI call sites are resolved at emission time into prebuilt call frames stored with the executable; emitted code loads the frame, fills operand pointers, and invokes a stable C trampoline. Rules: call-outs from sequential kernel context by default; a library that manages its own parallelism receives the pool from the context rather than nesting; errors propagate through the existing kernel status return. The pre-thunk emitter's `allow_runtime_calls` mechanism (still present in the code, disabled for region kernels precisely because the ABI lacks a context) is the feasibility proof.

This one capability dissolves several blockers at once: FFI custom calls (the LAPACK factorizations inside [jax#33666](https://github.com/jax-ml/jax/issues/33666)'s Newton loop), host callbacks (the progress-bar variant of [discussions#24501](https://github.com/jax-ml/jax/discussions/24501)), sort and fft (already runtime calls, just unreachable from kernels), and the tension between folding a loop and keeping library-quality kernels inside it. In-kernel copies lower to `llvm.memcpy` so folding never trades away the memcpy fast path.

### 4.3 The xtile region driver

A schedule-walking emitter that takes an outlined region computation and produces one MLIR function: members emitted in order through the existing xtile per-op emitters; `scf.while` and `scf.if` for control flow with loop-carried values as memrefs; tuple roots mapped to multiple result buffers; intermediates placed per 4.1. Reuses the existing tile analysis, the experimental region emitter's fusion-view flattening and constant-lifting mechanics, and the bufferization pipeline. The parity bar for replacing the legacy backend: bitwise-identical outputs at equal-or-better latency AND compile time on the full regression corpus. Compile time is a first-class bar because one large function is the known failure mode: LLVM middle-end cost grows ~N^1.8 in straight-line body size (2.17 s for a 128-step unrolled chain, against 73 ms at opt-level 0 with ~90% of the runtime win retained), so the driver owns emission-size discipline and opt-level tiering.

### 4.4 New emitters and capabilities (the build list)

1. Region driver (4.3): control flow, multi-output, scheduling, materialization. The largest single piece.
2. Multi-output tiling: `SymbolicTileAnalysis` currently accepts multiple roots only when one "real root" consumes the others; region shapes have independent roots. Under whole-shape single tiles there is no cross-root consistency problem, so the relaxation is contained.
3. Tiled reduce-window: window dimensions in the tiling space, the windowed affine map in propagation, a masked reduce emission; vectorization then comes free from the existing reduce lowering. Permanently fixes [jax#37465](https://github.com/jax-ml/jax/issues/37465). (An independent interim fix, an HLO rewrite of non-overlapping reduce-window to reshape+reduce, can ship earlier.)
4. Tiled gather (per output row, a dynamic-slice of the operand at a runtime index; scope to observed forms), reverse (normalize the negative stride the analysis already produces), dynamic-update-slice (write-side dual of the supported dynamic-slice; the in-place case must be gated on latency, not parity, because a naive lowering inserts a full-array self-copy per loop iteration that parity checks cannot see).
5. In-kernel sequential scatter (optional): a loop over index rows under whole-shape tiles; no tiling relation needed. Only if a workload demands the last measured gap (0.352 vs 0.238 ms on [jax#26145](https://github.com/jax-ml/jax/issues/26145)); small-scatter expansion already covers the class.
6. Mid-kernel call emission (4.2), shared by all emitters.
7. Parallel tiles: emit tile loops mapped onto the existing kernel workgroup dimensions, so one large kernel uses the intra-op pool. This is what lets kernel sizes grow without losing parallelism, replacing per-op partitioning for fused work.
8. Memcpy-aware copy emission.

### 4.5 The cost model

One function, three clients (kernel-boundary formation, per-edge materialization, library placement), replacing today's fixed constants. Benefit: saved thunk executions x ~50 ns x amplification (loop trip counts are usually known statically from `known_trip_count`), plus locality gains for tile-materialized edges. Costs: recompute (per shared producer: uses x producer cost, exact from the graph); inter-kernel overlap forfeited (subgraph total work minus critical path, capped by pool width); intra-op parallelism forfeited (would members partition standalone); register pressure (max live cut of the per-element expression vs the physical register file, deciding recompute-vs-materialize per edge); operand stream count vs the hardware prefetcher budget; working set vs cache capacities; library-vs-codegen deltas from a small calibrated table; code size; and a hard compile-time budget (member and basic-block caps as non-negotiable rails).

Every term is computable to order of magnitude from the graph plus a machine description (dispatch constant, ABI constant, register file, cache sizes, stream budget, pool width). The model's job is not precision; it is to avoid catastrophic corners, and every regression found during development is a corner a coarse model catches: hoisting copy-only regions (+633% and +1321% on two workloads; kernel ABI exceeded the dispatch saved), serializing independent once-per-call fusions (-13% on a 1B-parameter transformer), blocking loop hoisting to protect fusions for newer emitters (+550% on a high-trip-count scan), and a whole-module instruction-count gate that disabled hoisting on a 3,011-instruction model (+37%) because it was calibrated on benchmark slices smaller than production modules. Two standing lessons are encoded as constraints: gates must be per-run or per-region properties, never per-module (benchmark HLOs are often partial-model slices, so module-level thresholds do not transfer to production); and the model must be pinned by a continuously-run anchor suite plus production fold telemetry, or it rots.

### 4.6 What remains a boundary

Infeed/outfeed and collectives suspend execution and stay at thunk granularity. Everything else is eventually admissible.

## 5. Milestones

Each milestone states what gets measurably faster when it lands. M1 and M2 are independent and can run in parallel.

**M0: upstream the region-compilation prototype.** Region hoisting (generalizing `SmallWhileLoopHoistingPass`), small-scatter expansion, budget segmentation with amplification gating, the straight-line cap, and the thunk-generating minimum-size gate, sliced as two PRs (hoisting replaces the while pass and ports its tests; expansion stacks on it).
*Faster:* small stepping models with rolled loops, 3-4x ([jax#26145](https://github.com/jax-ml/jax/issues/26145): 1.53 to 0.36 ms, bitwise); loop-dominated scans up to 18x ([discussions#24501](https://github.com/jax-ml/jax/discussions/24501)); any small `lax.scan`/`while_loop` body drops from ~70-90 ns to ~25 ns per iteration of overhead. User guidance flips to "roll your loops on CPU": on jax 0.4.30 a rolled and an unrolled loop ran at the same speed; at head the unrolled form is 4.5x slower and ~12x slower to compile; with M0 the rolled form wins on both axes at every size. Cost: +14% compile on transformer-shaped models where per-layer bookkeeping folds (bounded, and excluded under the fast-compile preset); large-tensor models unaffected by construction.

**M1: runtime-context ABI and mid-kernel calls.** The context parameter, trampoline registry, FFI call frames, memcpy copies; sort and fft leave the boundary list.
*Faster:* stiff ODE solvers and anything with library calls inside hot loops ([jax#33666](https://github.com/jax-ml/jax/issues/33666), a 1.5-2x regression, recovers: the Newton loop folds with LAPACK called in-kernel); loops containing host callbacks (the progress-bar variant of [discussions#24501](https://github.com/jax-ml/jax/discussions/24501), reported at 13 to 25 s, stops splitting at every iteration); loops containing sort or fft fold for the first time.

**M2: the xtile region driver.** Control flow, multi-output, per-edge materialization; regions switch backends behind the existing experimental flag, then by default at the parity bar; `ComputationKernelEmitter` becomes dead code.
*Faster:* modest direct wins only (the legacy backend is already adequate at 32-element sizes; a measured ~35% on a synthetic dense region from vectorization and stack intermediates, more on medium shapes). The real value is what it unlocks: every M3 op lands once instead of twice, the protected-fusion conflict disappears (regions emit through the same modern emitters), and the legacy dependency is broken. Compile-time tiering (opt-level by region size) lands here.

**M3: op coverage on the modern path.** Tiled reduce-window, gather, reverse, dynamic-update-slice; optional in-kernel scatter. Each op ships with a bug-shaped test and a latency gate.
*Faster:* [jax#37465](https://github.com/jax-ml/jax/issues/37465)-class programs ~1.5x (the four scalar reduce-window kernels, ~12 us of ~30 us, become vectorized); [jax#26021](https://github.com/jax-ml/jax/issues/26021)-class dynamic-update-slice-heavy models (its 3.7x regression is DUS-dominated; the DUS work item plus M4 sizing target it); scatter-heavy loops gain the last ~1.5x if in-kernel scatter ships (0.35 to ~0.24 ms on [jax#26145](https://github.com/jax-ml/jax/issues/26145)).

**M4: cost-modeled boundaries and large kernels.** The quantitative model replaces the M0 gates, fusion sizing and region formation unify under it, per-edge materialization activates, parallel tiles let big kernels keep the pool, and the module gate and protection lists are deleted.
*Faster:* medium models: per-layer chains get properly sized kernels instead of threshold-shaped ones; trace-time-unrolled code (the [jax#26021](https://github.com/jax-ml/jax/issues/26021) shape) folds safely under the compile rails, measured 1.3-1.5x on synthetic unrolled chains; large-tensor fusions grow without serializing (today's byte gates forbid exactly the folds that would pay there). This is also where compile-time regressions become structurally impossible rather than gated: the budget is a model input.

**M5: delete the legacy emitters.** Inventory-driven: the region backend died at M2, the reduce-window fallback at M3, the `use_fusion_emitters=false` mode retires with a deprecation window; then `IrEmitter`, `IrEmitter2`, and `ComputationKernelEmitter` are removed and the pipeline's dual paths (e.g. the unconditional scatter-expansion branch) collapse.
*Faster:* nothing directly; this is the payoff in maintenance, single-path testing, and the ability to change the kernel ABI (M1's context, M4's scratch buffers) without double implementation.

## 6. Issue map

| issue | mechanism | closed by |
|---|---|---|
| [jax-ml/jax#26145](https://github.com/jax-ml/jax/issues/26145) (jaxley neuron sim, 4-5x) | thunk count in a 300-iteration loop; scatters fragment foldable runs | M0 (prototype measured: bitwise, at pre-thunk parity); M3 optional scatter adds ~1.5x |
| [jax-ml/jax discussions#24501](https://github.com/jax-ml/jax/discussions/24501) (scan + progress bar) | million-iteration scan next to hoisted invariants; callback splits the loop | M0 (6-18x, no-callback); M1 (callback variant) |
| [jax-ml/jax#33666](https://github.com/jax-ml/jax/issues/33666) (diffrax stiff, 1.5-2x) | LAPACK FFI + Python callback inside the Newton loop | M1 |
| [jax-ml/jax#37465](https://github.com/jax-ml/jax/issues/37465) (jnp.diff, 1.7-3.2x) | scalar legacy reduce-window emission | interim HLO rewrite any time; permanently M3 |
| [jax-ml/jax#26021](https://github.com/jax-ml/jax/issues/26021) (MJX mass matrix, 3.7x) | trace-time-unrolled joint loops, DUS-heavy, medium tensors | M3 (DUS) + M4 (sizing); library-side batching is a separate upstream conversation |

## 7. Risks

- **M2 parity:** the legacy emitter is battle-tested on control flow; the xtile driver replays that maturation. Mitigation: the bitwise-parity harness and regression corpus already exist; keep the legacy fallback flag one full release after parity.
- **M1 reentrancy:** library calls from kernels can nest pools or deadlock. Mitigation: sequential-context default, explicit pool handoff via the context, a nested-scan stress test, and the [jax#33666](https://github.com/jax-ml/jax/issues/33666) repro as the anchor.
- **Cost-model overfit:** anchors are finite. Mitigation: hard rails (compile budget, register cap) are non-negotiable regardless of model output; per-run-only gate properties; production fold telemetry as the drift alarm.
- **Coverage scope creep (M3):** op work is issue-driven with per-op measurement gates; an earlier coverage plan on this project was cancelled by measurement after one week of scoping, and its findings are reused here rather than relitigated.
- **Parallel tiles vs executor parallelism (M4):** one kernel must not be both internally parallel and concurrently overlapped naively; the cost model owns the either/or per kernel.

## 8. Glossary

- **Thunk:** the runtime's unit of execution, roughly one kernel launch per instruction or fusion. A thunk execution costs ~50 ns at small sizes (~18 ns dispatch + ~30 ns kernel-call ABI).
- **Amplification:** how many times per user-visible call a piece of code executes. While bodies (and anything nested under them) execute trip-count times; entry code executes once. Fixed overheads only matter where amplified, which is the central discriminator for every folding decision here.
- **Region:** a run of adjacent instructions outlined into a call and compiled as one kernel. On the dev branch these are formed by the region-hoisting pass and emitted by the legacy backend; after M2 they are xtile kernels, and after M4 "region" and "fusion" are the same construct.
- **Fold:** absorb an instruction into a region/kernel so it stops being a separately dispatched thunk.
- **Materialize / rematerialize:** whether a producer's value is written somewhere (buffer or stack) and reloaded, or recomputed at each consumer use. Elemental fusion rematerializes; dev-branch regions materialize to buffer slices; the end state chooses per edge.
- **Boundary:** an op that cannot join a kernel. Today: runtime-service ops plus coverage gaps. End state: only execution-suspending ops.
- **xtile:** the modern MLIR tiled-emission stack (`SymbolicTileAnalysis`, tile propagation, `EmitTiledComputation`, bufferization to stack allocations).
- **Fusion emitters:** the MLIR emitters for standalone fusions (`--xla_cpu_use_fusion_emitters`, default on at head), distinct from the xtile tiled path and from the legacy elemental emitters.
- **Sentinel:** the option value (byte threshold 0) that disables the dev-branch passes entirely; the upstream-equivalent baseline in every A/B measurement quoted here.
- **Partial-model slice:** a benchmark HLO cut from a larger program. Module-level properties measured on slices do not transfer to production modules; per-run properties do.
