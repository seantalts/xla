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
