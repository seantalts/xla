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

## 4. Component Specs

### 4.1 CPU Cost Model (kflops + parallelism)

### 4.2 MegaFusionPass (HLO pass)

### 4.3 Flag plumbing

### 4.4 Pipeline integration

## 5. Detailed Algorithm: MegaFusionPass

### 5.1 Preconditions & per-computation gate

### 5.2 Schedule-order region growing

### 5.3 Absorption predicate (per-op)

### 5.4 Emitting the kFusion instruction

### 5.5 Tuple/multi-output handling

### 5.6 Buffer aliasing & layout constraints

## 6. Data Structures

## 7. Worked Examples

### 7.1 Example 1: 35-joint mass-matrix (pure forward)

### 7.2 Example 2: Cholesky-like scan (while body)

## 8. File-by-File Change List

## 9. Test Plan

### 9.1 Unit tests

### 9.2 Integration / benchmarks

### 9.3 Regression guards

## 10. Rollout & Flag Semantics

## 11. Risks & Open Questions

## 12. v2 Roadmap

### 12.1 While-loop inlining

### 12.2 Parallelism-aware region splitting

### 12.3 Cost model refinement
