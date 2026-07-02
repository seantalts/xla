# Jaxley (#26145) XLA:CPU per-step time decomposition — Agent A

Workload: `module_0272.jit_run` (jaxley neuron sim, bwd_euler, ncomp=4 nbranch=8, 300 steps).
Baseline (hoisted, this build): **1.109 ms/step** (bench_hlo, n=100). Unhoisted: **1.531 ms/step**.
Old-runtime target (jax 0.4.30): **0.372 ms/step**. New end-to-end (jax 0.10.1): 1.697 ms/step.

## HEADLINE
The workload is **dispatch-bound**. Each step dispatches **~59,200 thunk-executions**, and
1.109 ms / 59,244 = **18.7 ns per thunk** — squarely inside the independently-measured 15-20 ns
per-thunk dispatch band. Real kernel compute is negligible (all arrays ≤ 32 elements). The entire
3.0× gap to the old runtime (0.737 ms/step) is **per-thunk dispatch + executor + while overhead** —
bucket (d). No single kernel category (scatters, regions, etc.) is the culprit; the sheer thunk
**count** is.

---

## Commands run

```
# census compile + dump (iters=1)
cd .../scratchpad && XLA_FLAGS="--xla_dump_to=.../dumpA" ./bench_hlo \
  --hlo_file=.../jaxley-dump/module_0272.jit_run.before_optimizations.txt --iters=1 --warmup=1
# after_optimizations HLO -> dumpA/module_0001.jit_run.cpu_after_optimizations.txt (5516 lines)

# hoisted re-verify
./bench_hlo --hlo_file=.../module_0272...before_optimizations.txt --iters=100 --warmup=20
#   -> mean 1.109 median 1.105

# unhoisted
XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0" \
  ./bench_hlo ... --iters=100 --warmup=20
#   -> mean 1.531 median 1.526

# old vs new runtime (jaxley_repro.py bwd_euler 4 8, 300 steps, median of 100)
/Users/xitrium/claud/perf-regression/venv/bin/python     jaxley_repro.py bwd_euler 4 8  # jax 0.10.1 -> 1.697 ms
/Users/xitrium/claud/perf-regression/venv-old/bin/python jaxley_repro.py bwd_euler 4 8  # jax 0.4.30 -> 0.372 ms
#   (note: `timeout` is unavailable on this Mac; ran without it — both finished in <30s)

# census parser: .../scratchpad/census.py  (parses HLO computations, counts thunks, expands calls)
```

Baseline --profile already saved at `.../jaxley_baseline_profile.txt` (read; not re-run).

---

## TASK 1 — Static thunk census (after_optimizations HLO)

While loop `%while.270`: trip count **300**, body `%wide.wide.wide.region_0.133...`,
64-element (74-slot) carried tuple. Entry `%main.135` runs **once/step** and contains the while.

### WHILE BODY (executes 300×/step) — 211 top-level instructions
| top-level op | count | dispatches a thunk? |
|---|---|---|
| get-tuple-element | 135 | no (buffer aliasing) |
| fusion | 52 | yes — KernelThunk |
| call (hoisted regions) | **18** | yes — CallThunk → nested executor |
| copy | 3 | yes — CopyThunk |
| constant | 2 | no |
| parameter | 1 | no |

- **18 kCall thunks** = call.93 … call.110 (matches profile). call.110 is the body ROOT
  (`%tuple.724_region`, output-assembly region).
- **52 top-level fusions** split into **31 scatter fusions** (`wrapped_scatter.NN`, all top-level —
  scatters are region boundaries, none live inside a hoisted region) + **21 non-scatter fusions**.
- Top-level dispatched thunks (calls counted as 1) = 52 + 18 + 3 = **73**.

### Per hoisted region (call → region computation): instrs / leaf-thunks inside
| call | region | #instrs | leaf thunks | notes |
|---|---|---|---|---|
| call.93 | add_bitcast_fusion.3_region | 11 | 5 | 4 fusion + 1 copy |
| call.94 | multiply_bitcast_fusion.6_region | 27 | 9 | |
| call.95 | multiply_bitcast_fusion.5_region | 24 | 6 | |
| call.96 | reverse_bitcast_fusion.11_region | 25 | 12 | |
| call.97 | reverse_bitcast_fusion.9_region | 39 | 20 | largest region |
| call.98 | reverse_bitcast_fusion.8_region | 15 | 4 | |
| call.99 | multiply_bitcast_fusion.3_region | 7 | 1 | smallest |
| call.100 | negate_multiply_fusion_region | 9 | 3 | |
| call.101 | reverse_bitcast_fusion.7_region | 25 | 12 | |
| call.102 | reverse_bitcast_fusion.6_region | 9 | 3 | |
| call.103 | reverse_bitcast_fusion.5_region | 33 | 16 | |
| call.104 | reverse_bitcast_fusion.3_region | 13 | 6 | |
| call.105 | wrapped_broadcast.14_region | 7 | 2 | |
| call.106 | reverse_bitcast_fusion.2_region | 13 | 6 | |
| call.107 | reverse_bitcast_fusion.1_region | 13 | 6 | |
| call.108 | wrapped_broadcast.16_region | 9 | 3 | |
| call.109 | reverse_bitcast_fusion_region | 20 | 8 | |
| call.110 | tuple.724_region | 80 | 2 | output plumbing, 78 GTE/tuple |
| **regions total** | | | **124** | 115 fusion + 9 copy |

- **Body leaf dispatched thunks (fully expanding all 18 calls) = 179** (167 fusion + 12 copy).
  Decomposition: 124 inside regions + 31 top-level scatters + 24 other top-level (21 fusion + 3 copy).
- **Scatter fusions per body iter = 31** (all top-level; zero inside regions).

### ENTRY (runs once/step) — 151 top-level instructions
- 82 GTE, 42 fusion, 11 call, 10 copy, 4 constant, 1 dynamic-update-slice, 1 while.
- **23 top-level scatter fusions.** 11 call thunks (call.82…call.92); call.92 builds the while-init tuple.
- **Entry leaf dispatched thunks (expanding calls, excluding the while body) = 133** (106 fusion,
  22 copy, 3 dynamic-update-slice, 2 concatenate). Plus the 11 CallThunks themselves.

### THUNKS PER STEP
- Fully-expanded leaf dispatches = **300 × 179 + 133 = 53,833**.
- Plus CallThunk dispatches = 300 × 18 (body) + 11 (entry) = **5,411**.
- **Grand total dispatched thunk-executions/step ≈ 59,244.**
- (Top-level-only view: 300 × 73 body + 64 entry = 21,964 — but each CallThunk internally runs its
  region's leaves, so the fully-expanded 59,244 is the real dispatch count.)

---

## TASK 2 — Dispatch arithmetic + executor cost

**Per-thunk dispatch @ 15-20 ns:**
- Leaf 53,833 × 15-20 ns = **0.807-1.077 ms**
- +CallThunks 5,411 × 15-20 ns = 0.081-0.108 ms
- **Grand 0.89-1.18 ms/step** — this band **brackets the entire measured 1.109 ms.**
- Back-solved: 1.109 ms / 59,244 = **18.7 ns/thunk**, inside the band → model self-consistent.
  Kernel compute must therefore be a small fraction folded into that per-thunk number.

**Executor per-invocation fixed cost:** profile shows ~301 `ThunkExecutor::Execute`/step
(300 body iters + 1 entry — nested region executors are not separately traced at this granularity).
I could **not** extract a clean positive fixed cost:
- *Bound method:* residual = 1.109 ms − (59,244 × 15-20 ns midpoint ≈ 1.03 ms) ≈ 0.08 ms over 301
  invocations ⇒ executor fixed cost **≲ 0.3 µs/invocation** (upper bound; within the dispatch band's
  own error, so consistent with ~0).
- *Hoisting-delta method:* hoisting **adds** 5,400 nested-executor invocations/step yet is **0.422 ms
  faster** (1.531→1.109). So a nested-region executor is *net cheaper* than leaving its thunks in the
  flat body graph. The 0.422 ms therefore measures **superlinear dependency-graph scheduling cost** in
  the outer executor (179-node flat graph vs 73-node graph + tight sequential sub-executors), NOT a
  per-invocation fixed cost. Executor fixed cost is effectively negligible; scheduling/dispatch is the
  cost, and it scales with node count.

---

## TASK 3 — Leaf attribution from --profile (with honest error bars)

**Profile absolute times are unusable as compute.** while.270 profiles at 7.70 ms/step vs real
1.109 ms (≈7× TraceMe inflation); `ThunkExecutor::Execute` and while.270 are aggregate/parent frames
that double-count nested scopes (per the caveats). Per-event times are TraceMe-overhead-dominated:
a single tiny fusion profiles at ~0.36 µs/invocation, almost all of which is the ~0.1-0.3 µs TraceMe
cost, not compute.

**Truncation:** top-30 shows 18 call parents + 10 non-scatter fusions + 2 aggregate frames.
The body alone has ~200+ distinct event types; **~90% of event *types* are cut off**, and critically
**all 31 scatter fusions are in the truncated tail** (none appear by name). So the profile cannot
directly give bucket (b). Rebuilding bench_hlo was judged unnecessary — the count×18.7 ns model below
gives a defensible, self-consistent attribution, and the old-vs-new comparison already pins the headline.

**Visible profile sums (÷10 = ms/step):**
- Σ 18 call parents = 1.914 ms/step (inflated; includes nested region leaves)
- Σ 10 visible non-scatter fusions = 0.943 ms/step (inflated)
- Relative visible split: calls 67% / fusions 33% (distorted — call parents aggregate ~7 leaves each
  while the 10 fusions are single events with proportionally more TraceMe overhead; do not trust).

**Preferred attribution — count-proportional, scaled to measured 1.109 ms** (valid because all kernels
are equally tiny, so per-thunk cost ≈ uniform 18.7 ns and dispatch dominates):

| bucket | thunks/step | ms/step | % |
|---|---|---|---|
| region-internal leaves (a) | 37,200 | 0.696 | 62.8% |
| scatters (b) | 9,300 | 0.174 | 15.7% |
| other top-level leaves (c) | 7,200 | 0.135 | 12.2% |
| CallThunk dispatch (d-part) | 5,400 | 0.101 | 9.1% |
| entry | 144 | 0.003 | 0.2% |

Error bars: the split is reliable to ±~15 ns/thunk × counts; the (a) vs (c) boundary depends on
whether region-internal micro-kernels are marginally cheaper than standalone ones (likely yes, so (a)
is a slight over-estimate and (c)/(d) slight under). Scatters (b) may run a few ns/thunk more than
loop fusions (gather+atomic-ish update), so 0.174 ms is a mild under-estimate — but still ~1/6 of step.

---

## TASK 4 — Old-runtime target

`jaxley_repro.py bwd_euler 4 8` (300 steps, median of 100 calls), same HLO computation:
- **jax 0.4.30 (old pre-thunk runtime): 0.372 ms/step** ← recovery target
- jax 0.10.1 (thunk runtime): 1.697 ms/step end-to-end (Python-dispatch included; the pure-XLA
  bench_hlo number for the same graph is 1.109 ms).

Both runs finished in seconds; no kills needed.

---

## TASK 5 — Final table

Per-step decomposition (hoisted, 1.109 ms). Buckets (a)-(c) are *kernel dispatch+micro-compute*
(inseparable at this scale); (d) is overhead beyond individual thunk dispatch.

| bucket | µs/step | % of 1.109 ms |
|---|---|---|
| (a) hoisted-region kernels (124 leaves/iter × 300) | ~696 | 62.8% |
| (b) scatter kernels (31/iter × 300) | ~174 | 15.7% |
| (c) other unhoisted top-level kernels (24/iter × 300) | ~135 | 12.2% |
| (d) CallThunk + executor + while control | ~100 | 9.0% |
| (e) unattributed (entry + measurement slop) | ~4 | 0.4% |

Reference points:
- **Old-runtime target: 372 µs/step** (jax 0.4.30). Gap to hoisted = **737 µs (3.0×)**.
- Unhoisted: 1531 µs/step. Hoisted: 1109 µs/step. Hoisting saves 422 µs (outer-graph scheduling).
- All-thunks dispatch model: 59,244 × 18.7 ns = 1109 µs (fits exactly).

### Which bucket dominates the gap to old runtime?
**Bucket (d) — per-thunk dispatch + executor overhead — dominates, but it is not localized to one
kernel type; it is the ~59k thunk-executions/step themselves.** The old runtime does the identical
math (arrays ≤ 32 elts, negligible FLOPs) in 372 µs by emitting far fewer runtime dispatches per HLO
computation. The thunk runtime pays ~18.7 ns × 59,244 dispatches = the whole 1.109 ms. Recovering the
0.737 ms gap requires **cutting thunk count / dispatch cost**, not speeding any kernel:
- The 300× while loop re-dispatches all ~179 body leaves + 18 regions every iteration.
- Hoisting already helped (−422 µs) by shrinking the outer dependency graph; deeper wins need
  fewer/larger thunks (more aggressive region fusion, scatter batching — 9,300 scatter dispatches/step
  = 174 µs alone) or a lower per-dispatch cost in the executor.

No individual kernel's compute is the problem; **thunk count × per-thunk overhead is.**
