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

## Interface

```cpp
// xla/service/cpu/cpu_cost_model.h

namespace xla::cpu {

struct CpuCost {
  double serial_cost;       // Abstract cycles on one core.
  double parallel_fraction; // In [0, 1].

  // Convenience: estimated wall-clock cost for `num_cores` cores.
  double EstimatedCost(int num_cores) const {
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

2. **Instruction scheduling**: Use `EstimatedCost(num_cores)` to order
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
  directly useful. Leaning toward abstract cycles, with `EstimatedCost`
  returning abstract cycles too, and letting the caller convert if needed.
- Do we need per-operand cost breakdown (to support scheduling decisions about
  which operand to prefetch)?  Probably not in v1.
- Should the weight config be per-microarchitecture or per-ISA-extension?
  Start with a single default config and refine later.
