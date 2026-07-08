# Agent E: loop re-rolling cost matrix (rolled vs unrolled vs batched), XLA:CPU

Spike question: does loop re-rolling pay on XLA:CPU, and does this branch
(SmallRegionHoistingPass + SmallScatterExpander, on by default) change the tradeoff?

- Host: Darwin arm64 (Apple Silicon), 2026-07-08.
- Binary: `/Users/xitrium/claud/xla-defrag/bazel-bin/xla/tools/bench_hlo` (prebuilt; no bazel).
- Flag settings:
  - **default** = our branch (both passes on).
  - **sentinel0** = `XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0"` ≈ upstream today (both passes disabled; cpu_compiler.cc treats 0 as the A/B escape hatch).
- Bench: `--iters=500 --warmup=50 --seed=42`, 2 timing runs per config (min reported); structure from a separate `--xla_dump_to` run at `--iters=1`.
- Artifacts (all under `.../scratchpad/reroll/`): generator `gen_hlo.py`, runner `run_matrix.py` (every command logged to `commands.log`), entry-op recount `recount.py`, raw results `results.json` / `matrix_out.jsonl`, HLO in `hlo/`, after-opt dumps in `dumps/<name>.<flagset>/`.

## Benchmark families

- **Family 1 (loop-carried, scan-like)**: state f32[4N], weight f32[4,4]; N steps of
  `x=slice/DS(state,4i); y=tanh(dot(weight,x)); state=DUS(state,y,4i)`, dependence chain state_0→state_N.
  `unrolled_N.hlo` (static slices + constant-offset DUS chain) vs `rolled_N.hlo` (while loop, carried (iter, state, weight)).
- **Family 2 (independent iterations)**: xs f32[N,4], weight f32[4,4]; per-row `tanh(x_i·W)`.
  `indep_unrolled_N` (N slices+dots+tanh, concat), `indep_rolled_N` (while, DUS into zero-init acc), `indep_batched_N` (`tanh(dot(xs,W))`, one [N,4]×[4,4] matmul).

**Correctness parity** (`--print_result --seed=42`, N=8): unrolled vs rolled **bitwise identical** (family 1); all three family-2 forms **bitwise identical** (row-major [1,4]×[4,4] dots preserve accumulation order, so no batched-vs-rowwise drift). default vs sentinel0 flags also bitwise identical (rolled_8).

## Results — Family 1 (loop-carried scan)

compile ms = min of 2 runs; run µs = mean over 500 iters (bench prints ms with 3 decimals → 1 µs quantization); "entry ops" = non-trivial top-level (thunk-level) instructions in after-opt ENTRY; "1-call" = whole program folded into a single `call` tagged `xla_cpu_small_call="true"`.

| N | flags | compile ms | run µs | entry ops | 1-call | while survived |
|---|---|---|---|---|---|---|
| unrolled 8 | default | 31.9 | 14 | 1 (call) | yes | n/a |
| unrolled 8 | sentinel0 | 32.5 | 15 | 25 (16 fusion + 8 dot + copy) | no | n/a |
| unrolled 32 | default | **205.3** | 19 | 1 | yes | n/a |
| unrolled 32 | sentinel0 | 68.6 | 24 | 97 | no | n/a |
| unrolled 128 | default | **2165.4** | 22 | 1 | yes | n/a |
| unrolled 128 | sentinel0 | 192.6 | 34 | 385 | no | n/a |
| unrolled 512 | default | 808.2 | 153 | 1537 | **no — 64KB gate** | n/a |
| unrolled 512 | sentinel0 | 1087.1 | 152 | 1537 | no | n/a |
| unrolled 512 | forcefold* | **29672.8** | 35 | 1 | yes | n/a |
| rolled 8 | default | 12.9 | 16 | 1 | yes | yes (trip 8) |
| rolled 8 | sentinel0 | 17.0 | 14 | 5 (while+tuple+2 copy+gte) | no | yes |
| rolled 32 | default | 12.4 | 16 | 1 | yes | yes |
| rolled 32 | sentinel0 | 16.8 | 18 | 5 | no | yes |
| rolled 128 | default | 12.7 | 18 | 1 | yes | yes |
| rolled 128 | sentinel0 | 17.0 | 29 | 5 | no | yes |
| rolled 512 | default | **12.7** | **27** | 1 | yes | yes |
| rolled 512 | sentinel0 | 16.9 | 47 | 5 | no | yes |

\* forcefold = `xla_cpu_small_while_loop_byte_threshold=16777216` (raises the straight-line region gate above the ~4MB aggregate footprint), 50 iters. Included to quantify what folding the 512-step chain would cost.

## Results — Family 2 (independent iterations)

| N | flags | compile ms | run µs | entry ops | 1-call | while survived |
|---|---|---|---|---|---|---|
| indep_unrolled 8 | default | 25.3 | 16 | 1 | yes | n/a |
| indep_unrolled 8 | sentinel0 | 34.8 | 14 | 17 | no | n/a |
| indep_unrolled 32 | default | **255.4** | 17 | 1 | yes | n/a |
| indep_unrolled 32 | sentinel0 | 63.6 | 22 | 97 | no | n/a |
| indep_unrolled 128 | default | **2924.1** | 24 | 1 | yes | n/a |
| indep_unrolled 128 | sentinel0 | 188.1 | 50 | 385 | no | n/a |
| indep_unrolled 512 | default | 793.8 | 93 | 1537 | **no — 64KB gate** | n/a |
| indep_unrolled 512 | sentinel0 | 691.1 | 93 | 1537 | no | n/a |
| indep_rolled 8 | default | 14.1 | 12 | 1 | yes | yes (trip 8) |
| indep_rolled 8 | sentinel0 | 17.7 | 11 | 6 (while+tuple+fusion+copy+gte) | no | yes |
| indep_rolled 32 | default | 14.6 | 12 | 1 | yes | yes |
| indep_rolled 32 | sentinel0 | 17.5 | 14 | 6 | no | yes |
| indep_rolled 128 | default | 13.7 | 15 | 1 | yes | yes |
| indep_rolled 128 | sentinel0 | 17.9 | 30 | 6 | no | yes |
| indep_rolled 512 | default | **13.9** | **23** | 1 | yes | yes |
| indep_rolled 512 | sentinel0 | 17.7 | 47 | 6 | no | yes |
| indep_batched 8 | default | 20.0 | 11 | 2 (fusions) | no (region < min_region_size=4) | n/a |
| indep_batched 8 | sentinel0 | 20.4 | 12 | 2 | no | n/a |
| indep_batched 32 | default | 15.4 | 18 | 2 | no | n/a |
| indep_batched 32 | sentinel0 | 14.7 | 16 | 2 | no | n/a |
| indep_batched 128 | default | 14.7 | 22 | 2 | no | n/a |
| indep_batched 128 | sentinel0 | 14.6 | 21 | 2 | no | n/a |
| indep_batched 512 | default | 14.6 | **22** | 2 | no | n/a |
| indep_batched 512 | sentinel0 | 14.8 | 21 | 2 | no | n/a |

## Scaling interpretation

**Compile time.**
- **Rolled is O(1) in N** under both flag settings: ~12.7 ms (branch) / ~17 ms (upstream), flat from N=8 to 512. Notably the branch compiles rolled *faster* than upstream (1 kernel module vs 5–6).
- **Unrolled under upstream** grows roughly linearly then worse: 32 → 69 → 193 → 1087 ms for N=8→512 (~1537 thunk-level ops, 1536 kernel functions at 512).
- **Unrolled under our branch** is superlinear and much worse at mid N: 32 → 205 → 2165 ms (N=8→128), because hoisting folds the whole chain into ONE LLVM function and single-function compile scales ~N^1.7–1.9. Force-folding N=512 costs **29.7 s** (37× upstream). The pass's 64KB aggregate-`bytes_accessed` gate (cpu_compiler.cc: `max(threshold, 1<<16)` for straight-line regions; while-loops gate on one body footprint, default 1KB knob) is what stops this at N=512 — the guardrail works, but N=32–128 sits under it and eats a **3×/11× compile regression** for a ~1.3–1.5× runtime gain.

**Runtime.** Bench floor is ~11–12 µs (everything at N=8, most of N=32, drowns in it — lean on N=128/512).
- **Rolled upstream** scales linearly above the floor at ~70–90 ns/iteration (4–5 thunk dispatches per body iteration): 14→18→29→47 µs. **Rolled on branch** cuts per-iteration cost ~3–4× (whole loop = one kernel; only the loop itself remains): 16→16→18→27 µs.
- **Unrolled upstream** is *worse* than rolled at every N here (15/24/34/152 vs 14/18/29/47 µs): 3N top-level thunk dispatches beat the loop-body dispatch cost. The historical "lax.scan is slow on CPU, so unroll at trace time" motivation does **not** reproduce at these shapes — unrolling already loses today on both axes.
- **Our branch flips nothing for rolled-vs-unrolled ordering (rolled already won) but widens it decisively**: rolled+branch is ≥ as fast as every unrolled config at runtime (27 vs 152 µs at N=512, 5.6×) while compiling 60–170× faster (12.7 ms vs 0.8–2.2 s). Rolled+branch strictly dominates. The right guidance for JAX users on this branch is: use `lax.scan`/`while_loop`; re-rolling pays on both axes.
- **BATCHED win**: over upstream forms at N=512 it's 2.2× vs rolled (21 vs 47 µs) and 4.4× vs unrolled (93 µs). Over branch-rolled it's ≈ nil at these shapes (22 vs 23 µs indep; branch-rolled family-1 27 µs) — once dispatch overhead is hoisted away, the remaining vector work ([N,4]×[4,4], ~16K flops) is too small for SIMD batching to matter. Expect batched to pull ahead again with larger inner dimensions where the matmul kernel's throughput (not dispatch) dominates.

**WhileLoopUnroller never fired**: all 16 rolled configs kept their `while` to after-opt HLO, with `known_trip_count` backend_config intact, even at trip count 8, under both flag settings. So no XLA-side re-unrolling contaminated the comparison.

## Caveats

- **Bench floor ~11–12 µs** (execute-call overhead): N=8 and largely N=32 rows are floor-dominated; deltas there are noise. Conclusions rest on N=128/512.
- **µs quantization**: bench prints mean_ms to 3 decimals, so run µs are ±0.5 µs quantized; compile ms is 2 samples/config (min shown), observed variance up to ~25% (e.g. unrolled_512 default 808 vs sentinel 1087 with identical after-opt structure — both unfolded; direction was consistent across samples but I would not read much into it).
- **Hoisting folds BOTH forms at N≤128**, so at small N the branch's unrolled-vs-rolled runtime comparison is one-giant-kernel vs one-giant-kernel; the differentiator there is compile time, and it is heavily in rolled's favor.
- **Tiny per-iteration compute by design** (4-wide rows): this isolates dispatch overhead but understates the batched/SIMD ceiling and understates how much real per-iteration compute would amortize upstream's per-thunk cost.
- **Compile-time regression risk of this branch on existing unrolled user code is real**: 3× at N=32, 11× at N=128 (both families), bounded only by the 64KB region gate. An instruction-count cap (or bytes-accessed-per-instruction profile) on straight-line regions is worth considering; the 29.7 s force-fold shows the failure mode if the gate were loosened.
- Single machine, single-threaded-ish tiny workloads, macOS arm64; absolute numbers will differ on server CPUs, ratios should be directionally stable.
