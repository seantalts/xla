# Agent C: proof-of-concept HLO hand-edits (jaxley module_0272)

Workload: /Users/xitrium/claud/perf-regression/jaxley-dump/module_0272.jit_run.before_optimizations.txt
Benchmark: bench_hlo --iters=100 --warmup=10 (mean/median ms per step). Baseline given: 1.097 ms.
All artifacts in this scratchpad dir. Noise band observed across repeats: ~±0.02 ms (~2%).

## Headline numbers

| Config | mean ms | median ms | vs baseline |
|---|---|---|---|
| POC-0 default flags (fresh run, seed 42) | 1.123 | 1.118 | baseline |
| POC-0 threshold=1e8, unedited | 1.108 | 1.105 | ~0 (noise) |
| POC-1b after-opt unedited, run_hlo_passes=false | 1.108 / 1.107 / 1.079 (3 runs) | 1.100 / 1.103 / 1.074 | reference for POC-1 |
| POC-1c full merge w/ duplication, 3 hot regions | 1.366 | 1.365 | **-24% (slower)** |
| POC-1c single-use merge (no dup), 3 hot regions | 1.084 / 1.094 / 1.085 | 1.079 / 1.088 / 1.079 | +1..2% (≈noise) |
| POC-1c single-use merge, all 29 regions | 1.104 / 1.106 / 1.098 | 1.095 / 1.106 / 1.097 | ~0 |
| **POC-2 scatters replaced, default flags** | **0.238** | **0.241** | **4.6x faster** |
| POC-2 scatters replaced, threshold=1e8 | 0.237 / 0.238 | 0.233 / 0.240 | same as default |

## POC-0: raised hoisting threshold on unedited module

- Command: `XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=100000000" ./bench_hlo --hlo_file=<before-opt> --iters=100 --warmup=10 --print_result --seed=42`
- Result: 1.108/1.105 vs default 1.123/1.118 — no change beyond noise.
- Parity: RESULT rows **bitwise identical** to default-flags seed-42 run (`poc0_default_seed42.txt` vs `poc0_bigthresh_seed42.txt`).
- Interpretation: the cost gate is not what limits region size today; scatter boundaries are. Letting the hoister take arbitrarily big regions buys nothing while scatters fragment the body.

## POC-1: merge fusions inside hoisted region kernels (semantics-preserving)

a. After-opt dump generated into `dumpC/` (`--xla_dump_to=... --iters=1`); module text copied to `poc1_afteropt_unedited.txt`.
b. Unedited after-opt text with `--run_hlo_passes=false`: 1.108/1.100, RESULT bitwise identical to the pipeline run. Confirms the hoisted call structure survives backend-only compilation.
c. Edit script: `poc1_merge_fusions.py` (parses HLO text, splices fused computations). Two modes:
   - Default: per root-tuple output, inline the entire producing chain into one giant kLoop fusion, **duplicating** shared producers into each consumer (as sanctioned by the POC brief). Applied to the 3 hottest call bodies (`%add_bitcast_fusion.3_region` = call.93, `%reverse_bitcast_fusion.9_region` = call.97, `%reverse_bitcast_fusion.5_region` = call.103) → `poc1_merged3.txt`, 10 giant fusions.
   - `--single-use`: only inline producers with exactly one user (no duplication, i.e. what a real fusion-merger pass would do) → `poc1_merged3_su.txt` (8 giants), `poc1_mergedall_su.txt` (ALL 29 region bodies, 24 giants).
   - Mid-fusion dynamic-update-slice was **accepted by the compiler and bitwise-correct**; the prepared `--dus-leaf` fallback was never needed.
d. Verification: every variant compiled with `--run_hlo_passes=false` and produced **bitwise-identical** seed-42 results (exact match, max abs diff 0).

Measurements:
- Duplicating merge: 1.366 ms (-24%). Profile: call.97 went 1.240 → 4.009 ms total (3000 execs) — elemental re-evaluation of duplicated gather/reverse/pad/DUS chains inside one fusion dominates. call.93: 1.297→1.374; call.103: 1.164→1.234.
- Single-use merge (fair version): 3 hot regions ≈1.088 avg vs ≈1.098 baseline avg → ~1% win, inside the noise band. Extending to all 29 regions: no additional win (≈1.103 avg).

**POC-1 verdict: NO. Intermediate materialization inside region kernels is not where the time goes.** Buffer-slice stores/loads between fusions within an already-single-kernel region are nearly free at these shapes (f32[32], f32[4,4], ...); merging buys ≤1-2% even duplication-free, and recompute-style merging actively hurts. A compiler pass to fuse harder inside regions is not worth building for this workload.

## POC-2: dissolve scatter boundaries (semantics-BREAKING structural ceiling)

a. Script `poc2_replace_scatters.py`: replaced **all 119 scatters** module-wide (they all execute within the step; most are in `%closed_call.131`, which gets inlined into the while body during optimization) with
   `result = add(operand, broadcast(add(convert(reduce_sum(updates)), convert(reduce_sum(indices)))))`
   Reduce-to-scalar helper computations (`%gfzaddredf32`, `%gfzaddreds32`) added at module scope (inserted after the stack-frame index section — inserting directly after the HloModule line breaks the parser). All three scatter inputs stay live. Output: `poc2_noscatter.txt` (0 scatters remain).
b. Compiled WITH passes, `XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=100000000 --xla_dump_to=.../dumpC2"`. The collapse **exceeded** the expected "~1-3 calls in the while body":
   - The pass outlined the **entire ENTRY into a single kCall** (`ROOT %call.29 = f32[1,301] call(), to_apply=%concatenate_bitcast_fusion_region`), whose 106-instruction region **contains the while loop itself**. The whole 300-iteration step is ONE kernel.
   - While-body instruction counts (after-opt): before = 211 total (18 call + 52 fusion + 135 GTE + 3 copy + 2 const + param); after = 193 total (**0 call**, 122 fusion, 53 GTE, 11 const, 4 copy, bitcast, tuple, param) — all living inside the single hoisted region.
   - Same result at **default flags** (`dumpC2def/`): still exactly 1 small_call containing the while. Scatter was the *only* boundary; regions containing control flow bypass the min-region-size gate (small_region_hoisting_pass.cc: `contains_control_flow`), and the whole-program region passes the bytes gate even at the 65536 floor.
c. Per-step: **0.238 ms mean / 0.240 median** (repeats 0.238, 0.237; default flags identical). Profile: exactly **1 thunk execution per step** (call.29, 0.209 ms in-kernel) vs ~301 thunk executions per step at baseline. Numerics intentionally wrong; no parity check.
d. Not needed — collapsed on the first attempt.

**POC-2 verdict: the structural ceiling of dissolving scatter boundaries is 1.097 → 0.238 ms/step (-0.86 ms, 4.6x).** Caveat: this overstates the win by the removed scatter compute itself (119 static scatters replaced by cheap reduce+broadcast+add; the loop executes ~47 of them per iteration) — being measured separately by another agent. Also note the substitute enabled ordinary fusion to do more (52 → 122 body fusions absorbing former scatter neighbors), so part of the gain is fusion across ex-scatter seams, not just thunk-count reduction.

## Key mechanism insight

Baseline spends its time on per-thunk dispatch (301 thunks/step; ThunkExecutor::Execute total 153.8 ms vs while.270 77.0 ms over 10 iters in the original profile), not on intra-kernel buffer traffic (POC-1) and not on region-size cost gating (POC-0). Everything funnels to: **make scatter stop being a region boundary** (single-kernel-emittable scatter, or scatter-aware region hoisting), which lets the existing pass swallow the entire loop at default settings.

## Files & commands log

Scripts:
- `poc1_merge_fusions.py` — region-body fusion splicer (modes: default duplicate-merge, `--single-use`, `--dus-leaf` [unused])
- `poc2_replace_scatters.py` — scatter → add/broadcast/reduce substitute

Edited HLO:
- `poc1_afteropt_unedited.txt` (copy of `dumpC/module_0001.jit_run.cpu_after_optimizations.txt`)
- `poc1_merged3.txt`, `poc1_merged3_su.txt`, `poc1_mergedall_su.txt`
- `poc2_noscatter.txt`

Run outputs: `poc0_default_seed42.txt`, `poc0_bigthresh_seed42.txt`, `poc1b_afteropt_baseline.txt`, `poc1_merged3_full.txt`, `poc1_merged3_profile.txt`, `poc1_merged3_su_full.txt`, `poc1_mergedall_su_full.txt`, `poc2_noscatter_run.txt`. Dumps: `dumpC/`, `dumpC2/`, `dumpC2def/`.

Commands (all from scratchpad dir):
```
./bench_hlo --hlo_file=<before-opt> --iters=100 --warmup=10 --print_result --seed=42            # POC-0 default
XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=100000000" \
  ./bench_hlo --hlo_file=<before-opt> --iters=100 --warmup=10 --print_result --seed=42          # POC-0 big threshold
XLA_FLAGS="--xla_dump_to=$PWD/dumpC" ./bench_hlo --hlo_file=<before-opt> --iters=1 --warmup=0   # POC-1a
./bench_hlo --hlo_file=poc1_afteropt_unedited.txt --run_hlo_passes=false --iters=100 --warmup=10 --print_result --seed=42
python3 poc1_merge_fusions.py poc1_afteropt_unedited.txt poc1_merged3.txt add_bitcast_fusion.3_region reverse_bitcast_fusion.9_region reverse_bitcast_fusion.5_region
python3 poc1_merge_fusions.py poc1_afteropt_unedited.txt poc1_merged3_su.txt --single-use <same 3 regions>
python3 poc1_merge_fusions.py poc1_afteropt_unedited.txt poc1_mergedall_su.txt --single-use ALL
./bench_hlo --hlo_file=poc1_merged*.txt --run_hlo_passes=false --iters=100 --warmup=10 --print_result --seed=42
python3 poc2_replace_scatters.py <before-opt> poc2_noscatter.txt
XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=100000000 --xla_dump_to=$PWD/dumpC2" \
  ./bench_hlo --hlo_file=poc2_noscatter.txt --iters=100 --warmup=10                              # POC-2 (0.238 ms)
XLA_FLAGS="--xla_dump_to=$PWD/dumpC2def" ./bench_hlo --hlo_file=poc2_noscatter.txt --iters=100 --warmup=10  # default flags, same 0.238
./bench_hlo --hlo_file=poc2_noscatter.txt --iters=100 --warmup=10 --profile                      # 1 thunk/step
```
