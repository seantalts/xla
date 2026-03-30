# XLA:CPU Cost Model Design

## Status: Draft

## Motivation

XLA:CPU lacks a real cost model. Today, optimization passes rely on a patchwork
of ad-hoc heuristics:

- **`InstructionFusion::IsExpensive`** — a boolean opcode whitelist. 50+ ops are
  "cheap" (add, multiply, reshape), 70+ are "expensive" (dot, reduce, sin).
  That's it. No consideration of shape, size, or element type. A 2×2 matmul and
  a 4096×4096 matmul are both just "expensive".
- **`DefaultCostModel`** in `parallel_task_assignment.cc` — computes a
  flops-to-bytes ratio and classifies ops into exactly two buckets (I/O-bound
  vs. compute-bound) with hard-coded thresholds. Returns a thread count
  directly, coupling the cost estimate to one specific consumer.
- **Shape-size thresholds** — scattered constants like "fuse if output < 16KB",
  "max 8 concat arguments", "max 5 reductions in a fusion". Each is a one-off
  heuristic tuned for a specific pass.
- **`HloCostAnalysis`** — computes raw FLOPs/bytes/transcendentals, but no CPU
  pass uses it directly for decision-making (the parallel task assigner wraps it
  but adds its own ad-hoc formulas on top).

Every compiler and query engine that takes performance seriously has a cost
model that maps operations to quantitative estimates. We should too.

## Prior art

### LLVM `TargetTransformInfo` (TTI)

LLVM's optimization passes query cost through `TargetTransformInfo`, a
target-specific interface with methods like `getInstructionCost`,
`getArithmeticInstrCost`, `getMemoryOpCost`, `getCastInstrCost`, etc. Each
returns an `InstructionCost` (a scalar integer with an "invalid" sentinel).

Key design properties:
- **Per-instruction, scalar cost.** No multi-dimensional cost — just one number
  representing "how expensive is this on the target?"
- **Target-parameterized.** Each backend (X86, AArch64, RISC-V) provides its own
  TTI implementation with different cost tables. A vector add on AVX-512 has a
  different cost than on NEON.
- **Consumer-driven cost kinds.** Callers specify a `TargetCostKind`
  (`TCK_RecipThroughput`, `TCK_Latency`, `TCK_CodeSize`, `TCK_SizeAndLatency`)
  to get different views of the same instruction. The vectorizer uses throughput;
  the unroller uses code size.
- **SIMD width is a separate concern.** TTI has `getRegisterBitWidth` and
  `getMaximumVF` — passes query vectorization width independently from
  per-instruction cost, then multiply.

**Takeaway:** A scalar cost per instruction, parameterized by target, is the
industry standard for single-threaded optimization passes. Multi-core
parallelism is handled at a higher level.

### Halide autoscheduler (Adams et al. 2019)

Halide separates the *algorithm* (what to compute) from the *schedule* (how to
compute it: tile sizes, parallelism, vectorization, storage order). The
autoscheduler uses a **learned cost model** (a small neural network) to predict
execution time for a given schedule.

Key design properties:
- **Input:** A featurized representation of the schedule — loop nest structure,
  tile sizes, memory footprint estimates, arithmetic intensity, parallelism
  level.
- **Output:** Predicted wall-clock runtime (a single scalar, in ms).
- **Parallelism is an input, not an output.** The autoscheduler explores
  different parallelism levels as part of the schedule search space. The cost
  model evaluates each candidate. It doesn't tell you *how parallel* an op is —
  it tells you how fast a *specific parallel schedule* would be.
- **Learned, not analytical.** Trained on actual runtimes from thousands of
  random schedules. This makes it accurate but opaque.

**Takeaway:** Separating "what to compute" from "how to execute it" is powerful.
Our cost model should describe the work (what), and let callers decide
parallelism (how).

### TVM / Ansor

TVM's autotuning framework uses cost models to guide search over operator
implementations:

- **XGBoost cost model**: Takes features extracted from a candidate schedule
  (loop structure, memory access patterns, vectorization width) and predicts
  relative runtime. Trained on profiled measurements.
- **Ansor** generates candidate programs via sketch-based search and uses the
  cost model to prune the search space before expensive hardware measurements.

Like Halide, the cost model evaluates *specific schedules*, not abstract ops.
Parallelism and tiling are part of the schedule being evaluated.

**Takeaway:** For autotuning, the cost model is a surrogate for measurement.
Our use case is different — we need fast, analytical estimates for compiler
passes, not search guidance. But the principle of featurizing ops by their
computational characteristics (FLOPs, memory footprint, arithmetic intensity)
is shared.

### Query engine cost models

#### PostgreSQL

PostgreSQL's optimizer estimates cost in terms of two scalars:
- **`startup_cost`** — cost before the first row is returned
- **`total_cost`** — cost to return all rows

Costs are in abstract units calibrated relative to a sequential page fetch
(= 1.0). Key cost parameters: `seq_page_cost` (1.0), `random_page_cost` (4.0),
`cpu_tuple_cost` (0.01), `cpu_operator_cost` (0.0025). These are
user-configurable via `SET`.

The optimizer uses table statistics (row counts, histograms, distinct value
counts) to estimate cardinalities, then applies per-operator cost formulas.
For example, a hash join costs `O(outer + inner)` to build/probe plus per-tuple
CPU cost scaled by estimated row counts.

**Takeaway:** Even a single scalar cost, well-calibrated, is enough to drive
good optimization decisions. Configurable cost weights (not hard-coded
constants) are essential for tuning to different hardware.

#### Apache Calcite

Calcite uses a **3-dimensional** cost: `(rowCount, cpu, io)`. Each relational
operator implements `computeSelfCost()` returning this triple. Plan comparison
uses a weighted combination, though in practice CPU cost dominates for in-memory
workloads.

**Takeaway:** Multi-dimensional cost is useful when different resources (CPU,
disk I/O, network) have fundamentally different characteristics. For XLA:CPU, we
have an analogous split between compute and memory bandwidth.

#### CockroachDB

CockroachDB uses a **single scalar** cost (float64), but its cost formulas
incorporate multiple factors: `cpuCostFactor`, `seqIOCostFactor`,
`randIOCostFactor`, `memCostFactor`, `latencyCostFactor`. It found that
collapsing to a scalar works fine as long as the formulas weight the factors
appropriately for the target hardware.

**Takeaway:** A scalar output doesn't mean single-factor reasoning. You can
model multiple bottlenecks internally and collapse to a scalar for comparison.

### Polyhedral compilers (Pluto / Polly)

Polyhedral compilers like Polly (in LLVM) don't use a traditional cost model.
Instead, they optimize a **dependence-distance objective**: minimize the
distance between dependent iterations in the transformed schedule. This is a
proxy for locality and parallelism without estimating absolute cost.

Parallelism falls out naturally: dimensions of the schedule with zero
dependence distance are parallelizable (Pluto's "coincidence" constraints).

**Takeaway:** For loop transformations, structural properties (dependences,
iteration space geometry) can substitute for quantitative cost. But for
instruction-level decisions (should we fuse these two ops?), you need numbers.

## What we're building

A cost model that takes an HLO instruction (opcode + shapes + attributes) and
returns a quantitative cost estimate. Not a boolean. Not a thread count. A
number that means "this much work."

### Core interface

```cpp
// xla/service/cpu/cpu_cost_model.h

namespace xla::cpu {

struct CpuCost {
  // Total work, in abstract cycles. This is the cost if executed on a
  // single core with no parallelism.
  double serial_cost;

  // Fraction of serial_cost that is parallelizable, in [0, 1].
  //   0 = entirely sequential (e.g. a scalar reduction's accumulation chain)
  //   1 = embarrassingly parallel (e.g. element-wise add)
  double parallel_fraction;

  // Estimated wall-clock cost for `num_cores` cores (Amdahl's law).
  double WallClock(int num_cores) const {
    double p = std::clamp(parallel_fraction, 0.0, 1.0);
    return serial_cost * ((1.0 - p) + p / std::max(1, num_cores));
  }
};

class CpuCostModel {
 public:
  virtual ~CpuCostModel() = default;

  // Cost of a single HLO instruction.
  virtual CpuCost GetCost(const HloInstruction& instruction) = 0;

  // Cost of a sequence of instructions (e.g. a fused computation).
  // Default: sums serial_costs, weighted-averages parallel_fraction.
  virtual CpuCost GetCost(absl::Span<const HloInstruction* const> instructions);
};

}  // namespace xla::cpu
```

Following LLVM TTI's lead, the cost model is target-parameterized: different
weight configs for different microarchitectures. Following PostgreSQL's lead,
the weights are configurable, not hard-coded.

The `parallel_fraction` field follows the Halide/TVM philosophy that the cost
model describes the work and lets callers decide parallelism — but we include
enough information for callers to make that decision without a second query.
This is the Amdahl's law formulation: `serial_cost` is the total work,
`parallel_fraction` says how much of it scales with cores.

### Why not just a scalar?

A single scalar cost would be simpler (CockroachDB gets away with it, LLVM TTI
is scalar). But our primary use case — parallel task assignment — needs to know
not just "how much work" but "how much of the work parallelizes." Today's
`DefaultCostModel` conflates these into a thread count; we want to separate
them so the same cost model serves scheduling, fusion, *and* parallelization.

That said, the `parallel_fraction` is a lightweight addition — one extra double
per instruction. If it turns out to be more trouble than it's worth, we can
always ignore it and use `serial_cost` alone.

### Input signals

Per instruction, the cost model consumes:

- **Opcode** — determines the cost formula.
- **Input shapes** (element type + dimensions) — determines FLOPs, memory bytes.
- **Output shape** — determines output bytes and whether the op is in-place.
- **Attributes** (e.g. convolution window, dot dimension numbers) — needed for
  non-trivial ops.

We intentionally do *not* take `num_cores` as input. That's the caller's
concern (via `WallClock`).

### Producing `serial_cost`

Re-use `HloCostAnalysis` to get raw FLOPs (`F`), transcendental count (`T`),
and bytes accessed (`B`). Combine with a weighted linear model:

```
serial_cost = w_f * F + w_t * T + w_b * B + w_0
```

Weights ship as a config struct, not hard-coded constants. The initial default
weights are calibrated to roughly match `DefaultCostModel`'s existing formula
(`1*F + 2*T + 10*B`) so we don't regress existing behavior.

Like LLVM TTI's per-target cost tables, we can later provide
microarchitecture-specific weight configs (e.g. different for AVX-512 vs. ARM
NEON, or for machines with different memory bandwidth).

### Producing `parallel_fraction`

Classify instructions into tiers:

| Category | `parallel_fraction` | Examples |
|---|---|---|
| Embarrassingly parallel | 1.0 | element-wise, broadcast, iota, pad |
| Data-parallel with reduction | ~0.9-0.95 | reduce, reduce-window |
| Partially parallel | ~0.5-0.8 | convolution, dot |
| Mostly sequential | ~0.0-0.3 | sort, while-loop body, scatter with conflicts |
| Non-parallelizable | 0.0 | rng, infeed, custom-call (unknown internals) |

For ops with data-dependent parallelism (like reduce), refine based on shape:

```
parallel_fraction = 1.0 - (reduced_elements / total_input_elements)
```

This is 0 for a full scalar reduction and ~1.0 for a reduce that only collapses
a single small dimension.

### Multi-instruction cost

For fused computations or instruction sequences:

```
total_serial_cost = sum(serial_cost_i)
total_parallel_fraction = sum(serial_cost_i * parallel_fraction_i) / total_serial_cost
```

Weighted average — the parallel fraction of combined work is dominated by
whichever sub-instruction contributes the most cost.

## Consumers

1. **Parallel task assignment** (`parallel_task_assignment.cc`): Replace
   `DefaultCostModel` and `SimpleCostModel`. Use:
   ```
   ideal_threads = serial_cost * parallel_fraction / min_cost_per_thread
   task_count = clamp(ideal_threads, 1, max_parallelism)
   ```

2. **Fusion decisions** (`cpu_instruction_fusion.cc`): Replace `IsExpensive`
   boolean with quantitative cost comparison. Instead of "is the producer
   expensive?", ask "does fusing reduce total cost?"

3. **Instruction scheduling**: Use `WallClock(num_cores)` to estimate critical
   path length.

## Comparison with existing XLA cost models

| Aspect | Base `HloCostAnalysis` | CPU `DefaultCostModel` | GPU `GpuHloCostAnalysis` | GPU Perf Model | **This proposal** |
|---|---|---|---|---|---|
| **Output** | FLOPs, bytes, seconds | Thread count (int) | FLOPs, bytes, IR size | Wall-clock time | `(serial_cost, parallel_fraction)` |
| **Parallelism model** | None | Binary (I/O vs compute) | None (perf model handles it) | Occupancy-based | Continuous fraction \[0,1\] |
| **Hardware params** | Rates (FLOP/s, B/s) | Hard-coded constants | Element-type profiles | Cache sizes, bandwidth, core count | Configurable weights |
| **Time model** | Bottleneck (max) | N/A | N/A | Overlapped (95%) | Amdahl's law |
| **Fusion support** | Sum of sub-ops | N/A | Utilization tracking, IR size limits | Runtime estimation | Weighted-average `parallel_fraction` |

## What's *not* in scope

- **Cross-device / network cost** — that's the auto-sharding cost model.
- **Memory placement** — the existing `OpCostManager` / MSA cost analysis
  handles this.
- **Cache modeling** — tempting but premature. We can add a
  `memory_pressure` field to `CpuCost` later if profiling shows it matters.
- **Learned cost model** — Halide/TVM show this can be very accurate, but
  requires training infrastructure and profiling data. Start analytical,
  consider learning later.

## Implementation plan

1. Add `CpuCost` struct and `CpuCostModel` interface in
   `xla/service/cpu/cpu_cost_model.h`.
2. Implement `DefaultCpuCostModel` that wraps `HloCostAnalysis` and applies the
   formulas above.
3. Write unit tests with known-shape instructions and expected cost ranges.
4. Wire into `ParallelTaskAssignment` behind a flag, validate against existing
   behavior.
5. Remove old `SimpleCostModel` / `DefaultCostModel` once validated.
6. Migrate `IsExpensive` callsites in CPU fusion to use quantitative cost.

## Open questions

- Should `serial_cost` be in abstract "cycles" or in seconds (requiring a
  target clock speed)? Abstract cycles are more portable; seconds are more
  directly useful. Leaning toward abstract cycles, with `WallClock`
  returning abstract cycles too, and letting the caller convert if needed.
- Do we need per-operand cost breakdown (to support scheduling decisions about
  which operand to prefetch)?  Probably not in v1.
- Should the weight config be per-microarchitecture or per-ISA-extension?
  Start with a single default config and refine later.
