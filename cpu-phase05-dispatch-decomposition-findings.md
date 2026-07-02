# Phase 0.5: Decomposing the Residual jaxley Regression — Measurements That Redirect the Plan

**Date:** 2026-07-02 · **Machine:** Apple M3 (darwin/arm64), single thread
**Context:** After `SmallRegionHoistingPass` (committed on `feat/cpu-small-region-hoisting`) folded the jaxley (JAX #26145) while-body into 18 region kernels, per-step latency was ~1.10 ms vs the old-runtime target of 0.372 ms — a residual 3.0× gap. Before investing in the planned four tiled-emitter features (multi-output/reverse/DUS/gather; see `cpu-tiled-region-coverage-design.md` on `design/cpu-stage1-region-compilation`), we profiled and ran hand-edit POCs to verify the target. **The plan changed completely.**

## Headline results

| configuration | ms/step | numerics |
|---|---|---|
| thunk runtime, hoisting disabled | 1.531 | baseline |
| thunk runtime + region hoisting (committed Stage-0) | 1.09–1.12 | bitwise = baseline |
| POC-1: hand-merged fusions inside region kernels | ~1.10 (no change) | bitwise = baseline |
| POC-2: scatters replaced by passthroughs (ceiling probe) | 0.238 | intentionally broken |
| **POC-3: scatters expanded (`--xla_cpu_use_fusion_emitters=false`), zero code changes** | **0.352** | **bitwise = baseline (seed-42)** |
| old runtime (jax 0.4.30, end-to-end `jaxley_repro.py`) | 0.372 | — |
| new JAX end-to-end (jax 0.10.1) | 1.697 | — |

**POC-3 recovers the entire regression** — it lands *below* the old-runtime number, semantics-preserving, using only in-tree machinery. The four-feature tiled-emitter plan is superseded for #26145.

## Finding 1 — the workload is thunk-granularity-bound, not codegen-bound

Static census of the optimized module (`--xla_dump_to`, after-opt HLO): the hot `while.270` runs **300 iterations/step**; its body executes ~73 thunks per iteration (18 hoisted region calls + 31 scatter fusions + ~24 other), plus ~144 entry thunks → **~22.4k thunk executions per step**. Average cost ≈ 50 ns per thunk execution (≈18 ns dispatch — matching our earlier slope-method calibration — plus ≈30 ns kernel ABI + compute on ≤32-element arrays). The old runtime spends ~7 ns per HLO op by emitting everything into one function.

*Correction recorded:* the decomposition subagent initially modeled hoisted region CallThunks as executing nested per-instruction thunks (59k thunks/step). That is wrong: `thunk_emitter.cc:663-700` emits `xla_cpu_small_call` kCalls as a **single kernel thunk** (RegionKernelEmitter or legacy ComputationKernelEmitter fallback), and the profile shows 301 `ThunkExecutor::Execute` invocations/step (one per while iteration + entry), not ~5,700. The corrected 22.4k count and ~50 ns/thunk average follow.

Verification steps: `bench_hlo --profile` per-thunk attribution (counts: 3,000 events per call name over 10 iterations = 300/iteration); executor invocation counting; census script over the after-opt dump (`phase05-artifacts/census.py`); code reading of the small-call emission path.

## Finding 2 — in-kernel codegen quality is NOT the lever (adversarial result)

The IR audit of hot region kernels found genuinely bad code: in `call.97_kernel`, ~90 of ~110 surviving loads re-read buffers the same function stored moments earlier (store-to-load forwarding defeated by a `!noalias` scope gap around memcpys); reverse and DUS segments are scalar; `call.103` (3rd hottest) has zero vector ops; each call pays ~28 arg-pointer loads + 19 tuple-pointer stores of ABI overhead. A register-SSA emitter looked worth 1.5–2.5× *on those kernel bodies*.

But the measured POC says the bodies don't matter at these shapes: hand-merging fusions inside the region computations so intermediates rematerialize in registers (semantics-preserving, bitwise-verified) moved end-to-end latency by **~1% (noise)**; a variant that duplicated shared chains was **24% slower** (elemental recompute of gather/reverse chains dominates). At ≤32-element shapes, intermediate materialization inside a kernel is essentially free; the cost is *around* kernels (dispatch + ABI), not *inside* them.

This kills the codegen-quality justification for the tiled-emitter path on this workload. (Full IR evidence: `phase05-artifacts/agentB_ir_inspection.md`; POC data: `agentC_poc_edits.md`.)

## Finding 3 — scatter is the only thing preventing whole-program collapse

Replacing all scatters with shape-correct passthroughs (semantics intentionally broken) and recompiling let `SmallRegionHoistingPass` outline the **entire entry computation — the 300-iteration while loop included — into ONE kernel**: 1 thunk/step vs ~22.4k. Latency: **0.238 ms/step (4.6×)**. This happens at *default* flags: regions containing control flow already bypass the min-region-size gate, and the whole program's buffers fit under the 64 KB byte gate. Scatter was the only boundary op present.

Raising the hoisting byte threshold alone (no edits) changed nothing — the cost gate is not the limiter; boundaries are.

## Finding 4 — the discovery: scatter expansion already dissolves the boundary, in-tree, today

Why scatter is a boundary: legacy `IrEmitter::HandleScatter` is `Unimplemented` (`ir_emitter.cc:1896`), so the hoisting pass lists `kScatter` (and anything whose called computations contain one) as unavailable.

But the CPU pipeline already contains `ScatterExpander(ScatterExpander::kEliminateAllScatters)` (`cpu_compiler.cc:927-929`) — the reference lowering of scatter into while+gather+dynamic-update-slice, all ops the legacy region emitter handles natively. It is bypassed only when the MLIR fusion-emitter scatter is enabled. Setting the existing debug flag flips it back on:

```
bench_hlo --hlo_file=jaxley-dump/module_0272.jit_run.before_optimizations.txt \
  --iters=100 --warmup=10 --print_result --seed=42 \
  --xla_cpu_use_fusion_emitters=false
```

Result: **0.352 ms/step, bitwise-identical output to baseline (seed-42)**. Structure of the optimized module: entry = **1 call instruction** (one kernel, one thunk per step); 0 scatters; 55 while loops (1 main + 54 expanded scatters, matching the 54 scatters in the baseline module) all emitted as native LLVM control flow inside the single function; 191 fusions, 111 DUS, 91 gathers — all region-internal.

The 0.352 vs 0.238 delta (~114 µs) is the real scatter compute (54 nested scalar while loops per step) that POC-2's passthroughs faked away.

## Go-forward plan: selective scatter expansion

`--xla_cpu_use_fusion_emitters=false` is a global hammer (disables the MLIR fusion emitters everywhere) — fine as a user workaround for small scatter-heavy models today, wrong as a shipped default. The shippable form is a **selective** scatter expansion:

- **What:** expand scatters whose operand+updates+indices byte volume is small (aligned with the hoisting byte gate) at the existing pipeline position (`cpu_compiler.cc:921-929` sandwich — before post-expansion simplification, fusion, and copy insertion, so all downstream machinery processes the expanded form normally). Everything else (hoisting, legacy region emission, control-flow-in-region) already works.
- **Key risk:** an expanded scatter that hoisting does *not* later swallow becomes a while *thunk* executing per-index-row body thunks — much slower than the dedicated MLIR scatter kernel. Mitigations: conservative size predicate; measure on scatter-heavy non-hoistable workloads; v2 predicate could dry-run the hoisting pass's own region-eligibility analysis ("would this scatter fragment an otherwise-hoistable region").
- **First verification task:** confirm the collapse still happens with fusion emitters ON + expansion forced (POC-3 had them off) — hoisting is HLO-level and region kernels use legacy emission internally regardless, so it should, but measure it.
- **Gates:** bitwise parity on jaxley; ≤0.4 ms/step; no regression on a large-scatter benchmark (its scatters must not match the predicate); existing hoisting tests green.

Residual value of the tiled-emitter four-phase plan: not justified by #26145 anymore (0.352 already beats the old runtime). Revisit only if other workloads demand it; the phases remain documented as an appendix in the coverage design doc.

## Scope notes

This addresses #26145 (jaxley). #37465 (reduce-window) and #33666 (LAPACK FFI + callbacks) have different root causes (see `cpu-small-model-regression-findings.md`) and are unaffected by this plan.

## Method log

Three parallel subagent investigations + inline verification, all on `feat/cpu-small-region-hoisting` @ `3a1b70020e`, `bench_hlo` (dev tool, uncommitted), 100 timed iterations, warmup 10:

1. Decomposition agent: static thunk census, dispatch arithmetic, venv-old (jax 0.4.30) vs venv (0.10.1) end-to-end via `jaxley_repro.py` → `phase05-artifacts/agentA_decomposition.md` (with the CallThunk-count correction noted above).
2. IR audit agent: per-kernel `.ir-no-opt.ll`/`.ir-with-opt.ll` diffing for `call.93/.97/.103` + reference standalone fusion → `phase05-artifacts/agentB_ir_inspection.md`.
3. POC-edit agent: threshold raise (POC-0), intra-region fusion merging via `poc1_merge_fusions.py` (POC-1), scatter passthroughs via `poc2_replace_scatters.py` (POC-2) → `phase05-artifacts/agentC_poc_edits.md`.
4. Inline (orchestrator): CallThunk emission-path verification; `HandleScatter`/`ScatterExpander` code reading; POC-3 run, parity diff, and structural counts on the after-opt dump.
