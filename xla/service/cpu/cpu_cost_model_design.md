# XLA:CPU 2D Cost Model Design

## Status: Draft

## Motivation

XLA:CPU needs a cost model that provides per-instruction cost estimates for use
in scheduling, fusion decisions, and parallel task assignment. Today these
concerns are scattered:

- `HloCostAnalysis` computes raw FLOPs / bytes / transcendentals.
- `DefaultCostModel` in `parallel_task_assignment.cc` combines those into a
  single heuristic that *also* decides parallelism, coupling two concerns.
- The `OpCostManager` framework (`cost_modelling/op_cost.h`) provides a clean
  metric interface but currently only targets memory-space-assignment latency
  estimates.

None of these answer the question a scheduler or autotuner actually asks:
**"How long will this instruction take on a machine with N cores?"** To answer
that, you need two things: (1) the total work, and (2) how much of that work
can run in parallel.

## Proposal: Return Two Scalars

```cpp
struct CpuCost {
  // Total work, in abstract cycles. This is the cost if executed on a
  // single core with no parallelism.
  double serial_cost;

  // Fraction of serial_cost that is parallelizable, in [0, 1].
  //   0 = entirely sequential (e.g. a scalar reduction's accumulation chain)
  //   1 = embarrassingly parallel (e.g. element-wise add)
  double parallel_fraction;
};
```

A caller can derive wall-clock cost for a specific machine as:

```
wall_clock = serial_cost * ((1 - parallel_fraction) + parallel_fraction / N)
```

where `N = min(available_cores, max_useful_threads(instruction))`. This is
just Amdahl's law. The nice thing is that the cost model itself never needs to
know how many cores you have.

### Why not two separate models?

You asked: *"Should we just have a scalar cost model and a separate scalar
parallelizability model?"*

That's a legitimate alternative. Here's the trade-off:

| | Two-scalar struct | Separate models |
|---|---|---|
| **Co-reasoning** | The parallel fraction can depend on the same analysis that produces the cost (e.g. the flops-to-bytes ratio already computed). Bundling avoids redundant work. | Each model is simpler in isolation. |
| **Composability** | A single `CpuCostModel::GetCost(instruction) -> CpuCost` call is easy to plumb. | Two calls, but callers that only need one dimension can ignore the other. |
| **Testing** | One test per op covers both dimensions. | More focused unit tests per concern. |
| **Extensibility** | Adding a third dimension later (e.g. memory-bandwidth pressure) means changing the struct everywhere. | Adding a third model is just a new class. |

**Recommendation:** Use the two-scalar struct. The two quantities are
fundamentally entangled (you can't reason about parallelizability without
understanding the work breakdown), and the primary consumer always needs both.
If a third dimension is needed later, we can evolve `CpuCost` to a small
struct with named fields - that's a minor, mechanical refactor.

## Comparison with existing cost models

### Base `HloCostAnalysis` (all backends)

`HloCostAnalysis` (`xla/service/hlo_cost_analysis.h`) is a DFS visitor that
computes raw metrics per instruction: FLOPs, transcendental count, bytes
accessed, and an "optimal seconds" estimate. Time is computed via a
**bottleneck model**:

```
optimal_seconds = max(flops / flops_per_sec,
                      bytes / bytes_per_sec,
                      transcendentals / transcendentals_per_sec)
```

This assumes perfect overlap between compute and memory — whichever is slower
dominates. It has no notion of parallelism or core count. The `Properties`
class uses a hybrid fast-path / hash-map lookup that was documented as "the
most impactful single optimization we were able to make to GPU compilation
time", so any new model should avoid regressing cost-analysis overhead.

**Relationship to this proposal:** We reuse `HloCostAnalysis` to extract raw
FLOPs/bytes/transcendentals as inputs to our `serial_cost` formula. We do not
replace it.

### CPU `DefaultCostModel` (today)

The current CPU cost model in `parallel_task_assignment.cc` computes a
flops-to-bytes ratio and classifies instructions into two buckets:

- **I/O-bound** (ratio ≤ 1.0): Uses `bytes_accessed` as cost, caps parallelism
  at `sqrt(num_cores)`, and uses 256 KB (L2 cache size) as the minimum cost per
  thread.
- **Compute-bound** (ratio > 1.0): Uses `1*flops + 2*transcendentals +
  10*bytes_accessed` as cost, allows full parallelism, and uses 100K cycles
  (~50 µs at 2 GHz) as minimum cost per thread.

This is essentially a hand-rolled Amdahl's law with only two discrete
parallelism levels. The limitations:

1. **Coupled concerns** — the model returns a thread count directly, so it
   can't be reused by the scheduler or fusion heuristics without re-deriving
   the cost.
2. **Binary classification** — ops are either "I/O bound" or "compute bound"
   with nothing in between. A matmul that's slightly memory-bound gets
   sqrt-capped parallelism, while one that's slightly compute-bound gets full
   parallelism.
3. **Hard-coded constants** — the weights (1, 2, 10), the 256 KB threshold,
   and the 100K cycle minimum are baked in with no per-target tuning.

Our proposal replaces this with a continuous `parallel_fraction` and
configurable weight parameters.

### GPU `GpuHloCostAnalysis`

`GpuHloCostAnalysis` (`xla/service/gpu/model/gpu_hlo_cost_analysis.h`)
extends `HloCostAnalysis` with GPU-specific refinements:

- **Element-type-aware FLOPs**: Uses precomputed `HloOpProfiles` mapping
  `(opcode, element_type)` → FLOPs-per-element (e.g. `exp` on f32 costs more
  than `add` on f32). Default is 3 FLOPs/element for unknown ops.
- **Fusion utilization analysis**: Forward-traverses fused computations to track
  how many times each sub-instruction is emitted and what fraction of each
  operand is actually read. This prevents over-counting in large fusions.
- **IR size tracking**: Estimates generated code size to prevent fusions that
  would exceed ~10K IR instructions and blow up compile time.
- **Collective operation costs**: Models ring/tree algorithms with
  device-count-aware scaling ratios (e.g. `num_ranks / (2*(num_ranks-1))` for
  ring all-reduce).

**What we can learn from it:**
- Per-element-type profiling is valuable. Our initial version uses uniform
  weights, but we should plan for type-aware `serial_cost` refinements (e.g.
  f64 div is much more expensive than f32 add on x86).
- Fusion utilization analysis matters. Our multi-instruction `GetCost` should
  eventually account for operand reuse rather than naively summing.

### GPU Performance Model

The GPU performance model (`gpu_performance_model_base.h`) goes a step further,
converting costs into wall-clock time estimates using detailed hardware
parameters:

- **Cache hierarchy**: Models L1 (8× speedup) and L2 (2.5× speedup) relative
  to DRAM bandwidth, with size-based tiering.
- **Compute-memory overlap**: Uses an empirically-calibrated 95% overlap
  constant (not 100%) to account for real-world synchronization overhead:
  ```
  exec_time = compute_time + memory_time
            - min(compute_time, memory_time) × 0.95
  ```
- **Launch overhead**: Adds per-kernel launch cost (1 µs normal, 5 µs for NCCL
  kernels).
- **Occupancy-aware bandwidth**: Scales effective bandwidth based on the number
  of active thread blocks vs. available SMs.

This level of hardware modeling is appropriate for GPU where the execution model
(warps, SMs, shared memory) is rigid and well-characterized. For CPU, the
execution model is more flexible (out-of-order cores, OS scheduling, varying
cache sizes), so we intentionally start simpler with abstract cycles and defer
cache modeling.

### Summary comparison

| Aspect | Base `HloCostAnalysis` | CPU `DefaultCostModel` | GPU `GpuHloCostAnalysis` | GPU Perf Model | **This proposal** |
|---|---|---|---|---|---|
| **Output** | FLOPs, bytes, seconds | Thread count (int) | FLOPs, bytes, IR size | Wall-clock time | `(serial_cost, parallel_fraction)` |
| **Parallelism model** | None | Binary (I/O vs compute) | None (handled by perf model) | Occupancy-based | Continuous fraction \[0,1\] |
| **Hardware params** | Rates (FLOP/s, B/s) | Hard-coded constants | Element-type profiles | Cache sizes, bandwidth, core count, FPU count | Configurable weights |
| **Time model** | Bottleneck (max) | N/A | N/A | Overlapped (95%) | Amdahl's law |
| **Fusion support** | Sum of sub-ops | N/A | Utilization tracking, IR size limits | Runtime estimation | Weighted-average `parallel_fraction` |
| **Cache modeling** | None | 256 KB L2 assumption | None | L1/L2 tiered | Deferred |

## Interface

```cpp
// xla/service/cpu/cpu_cost_model.h

namespace xla::cpu {

struct CpuCost {
  double serial_cost;       // Abstract cycles on one core.
  double parallel_fraction; // In [0, 1].

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
  // Default implementation sums serial_costs and takes the min
  // parallel_fraction, which is conservative.
  virtual CpuCost GetCost(absl::Span<const HloInstruction* const> instructions);
};

}  // namespace xla::cpu
```

### Input signals

The cost model consumes, per instruction:

- **Opcode** - determines the cost formula.
- **Input shapes** (element type + dimensions) - determines FLOPs, memory bytes.
- **Output shape** - determines output bytes and whether the op is in-place.
- **Attributes** (e.g. convolution window, dot dimension numbers) - needed for
  non-trivial ops.

We intentionally do *not* take `num_cores` as input. That's the caller's
concern.

### Producing `serial_cost`

Re-use `HloCostAnalysis` to get raw FLOPs (`F`), transcendental count (`T`),
and bytes accessed (`B`). Combine with a simple linear model similar to
`DefaultCostModel`:

```
serial_cost = w_f * F + w_t * T + w_b * B + w_0
```

Weights can be target-specific (e.g. different for AVX-512 vs. ARM NEON) and
will ship as a small config struct, not hard-coded.

### Producing `parallel_fraction`

Classify instructions into tiers:

| Category | `parallel_fraction` | Examples |
|---|---|---|
| Embarrassingly parallel | 1.0 | element-wise, broadcast, iota, pad |
| Data-parallel with reduction | ~0.9-0.95 | reduce, reduce-window (parallel across output elements, serial accumulation within each) |
| Partially parallel | ~0.5-0.8 | convolution (parallel across batch/output-channel, serial within kernel), dot |
| Mostly sequential | ~0.0-0.3 | sort, while-loop body, scatter with conflicts |
| Non-parallelizable | 0.0 | rng, infeed, custom-call (unknown internals) |

For ops with data-dependent parallelism (like reduce), we can refine the
estimate based on shape: a reduce over a large dimension with a small output is
less parallelizable than a reduce over a small dimension with a large output.

A simple heuristic for reduce:

```
parallel_fraction = 1.0 - (reduced_elements / total_input_elements)
```

This is 0 for a full scalar reduction and ~1.0 for a reduce that only collapses
a single small dimension.

### Multi-instruction cost

For fused computations or instruction sequences, the default aggregation is:

```
total_serial_cost = sum(serial_cost_i)
total_parallel_fraction = sum(serial_cost_i * parallel_fraction_i) / total_serial_cost
```

This is a weighted average - the parallel fraction of the combined work is
dominated by whichever sub-instruction contributes the most cost. This is more
accurate than taking the min.

## Consumers

1. **Parallel task assignment** (`parallel_task_assignment.cc`): Replace
   `DefaultCostModel` and `SimpleCostModel`. Instead of the current ad-hoc
   flops-to-bytes classification, use:
   ```
   ideal_threads = serial_cost * parallel_fraction / min_cost_per_thread
   task_count = clamp(ideal_threads, 1, max_parallelism)
   ```

2. **Instruction scheduling**: Use `WallClock(num_cores)` to order
   instructions on the critical path.

3. **Fusion decisions**: Compare `GetCost(fused)` vs `sum(GetCost(unfused))` to
   evaluate whether a fusion is profitable.

## What's *not* in scope

- **Cross-device / network cost** - that's the auto-sharding cost model.
- **Memory placement** - the existing `OpCostManager` / MSA cost analysis
  handles this well.
- **Cache modeling** - tempting but premature. We can add a
  `memory_pressure` field to `CpuCost` later if profiling shows it matters.

## Implementation plan

1. Add `CpuCost` struct and `CpuCostModel` interface in
   `xla/service/cpu/cpu_cost_model.h`.
2. Implement `DefaultCpuCostModel` that wraps `HloCostAnalysis` and applies the
   formulas above.
3. Write unit tests with known-shape instructions and expected cost ranges.
4. Wire into `ParallelTaskAssignment` behind a flag, validate against existing
   behavior.
5. Remove old `SimpleCostModel` / `DefaultCostModel` once validated.

## Open questions

- Should `serial_cost` be in abstract "cycles" or in seconds (requiring a
  target clock speed)? Abstract cycles are more portable; seconds are more
  directly useful. Leaning toward abstract cycles, with `WallClock`
  returning abstract cycles too, and letting the caller convert if needed.
- Do we need per-operand cost breakdown (to support scheduling decisions about
  which operand to prefetch)?  Probably not in v1.
- Should the weight config be per-microarchitecture or per-ISA-extension?
  Start with a single default config and refine later.
