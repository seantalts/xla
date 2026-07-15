# Experiment brief: calibrating the region-hoisting gates on a broad CPU benchmark corpus

**Audience:** an agent/engineer picking this up fresh. Everything needed is in this file plus the linked docs.
**Date:** 2026-07-15
**Code under test:** [`feat/cpu-small-region-hoisting`](https://github.com/seantalts/xla/tree/feat/cpu-small-region-hoisting) at [`819cf1576d`](https://github.com/seantalts/xla/commit/819cf1576d) (region hoisting with budgeted segmentation + loop-amplification gating, plus small-scatter expansion).
**Background reading (short):** [pass design doc](https://github.com/seantalts/xla/blob/design/cpu-stage1-region-compilation/cpu-small-region-hoisting-pass-design.md), especially "Budgeted segmentation" and open questions 1, 2, and 4.

## The three questions to answer with data

1. **Module-size gate: presumed WRONG; measure whether local gates suffice instead.** A module-level instruction-count skip was considered and is deprioritized for a structural reason: much of the benchmark corpus is *partial-model* HLO (a slice of a larger program), so module-level thresholds calibrated on the suite will not transfer to production. A gate that looks harmless on benchmarks (it never fires there) will fire on the full-size production modules that contain the same hot loops, and starve them. Run-level criteria (byte budget, member cap, amplification) are composition-invariant: a run behaves the same whether it sits in a 100-instruction slice or a 30k-instruction model, which is the property a gate needs when the corpus is made of slices. Known module counts for orientation only (total post-opt instructions incl. fusion bodies): 24501 repro = 72 (6-18x win), jaxley = 4,434 (4.2x win), Gemma3 1B = 20,776 (runtime wash, +14% compile), TORAX ~29k. What phase A should actually answer: do the local gates alone keep every big-module case at wash-or-better runtime with acceptable compile cost? Build the counterexample regardless (a large module whose runtime is dominated by one hot small loop, e.g. the 24501 scan embedded in ~10k instructions of medium dot/elementwise chains): under local-gates-only it must WIN, which is the anti-module-gate evidence; if compile cost on big modules turns out unacceptable, the answer is per-region compile budgeting or preset gating, calibrated on production fold-rate telemetry, not a benchmark-derived module threshold.
2. **Per-run straight-line member cap** (open question 1). The compile blowup (~N^1.8, 2.17 s at a 128-step unrolled chain) happens in SMALL modules, so the module gate cannot replace this. Implement the cap as a second cut criterion in the existing segmentation (see "Implementing the cap" below), then calibrate the value: candidates {32, 64, 96, 128}. The cost of a low cap is extra per-iteration dispatch when a large amplified region splits (jaxley's largest body region is ~80 members; measure its runtime delta at each cap). The cost of a high cap is residual compile time on scalar chains.
3. **Kill non-amplified straight-line folding entirely** (open question 4). Every measurement of it so far is zero or negative: the frag microbenchmark was flat, Gemma3 straight-line segments were -13%, unrolled chains are the compile blowup. If the corpus confirms no straight-line once-per-call fold ever wins, apply the amplification keep-rule to unsegmented runs too, which deletes question 2's entry-side case and most of the compile-time risk class.

## Environment

- Worktree: `/Users/xitrium/claud/xla-defrag` (branch above). Work on a NEW branch off it; do not rewrite existing commits.
- Build/test incantation (Apple Silicon quirks are mandatory):
  `bazel build -c opt --config=macos --config=macos_arm64 --macos_sdk_version=15.4 --copt=-Wno-error --host_copt=-Wno-error //xla/tools:bench_hlo`
  (same flags for `bazel test`). Bench binary lands at `bazel-bin/xla/tools/bench_hlo`.
- `bench_hlo` usage: `--hlo_file=X --iters=N --warmup=M [--print_result --seed=42] [--profile]`. Prints `Compiled ... in X ms` and `BENCH: ... mean_ms=...`. XLA_FLAGS honored.
- The pass pair is ON by default. The upstream-equivalent baseline is the sentinel:
  `XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0"`.
- Measurement discipline (non-negotiable, this machine is shared): check `uptime` first and do not trust absolute numbers above load ~5; always interleave default/sentinel reps (D,S,D,S) and report both; every runtime comparison carries a bitwise parity check (`--print_result --seed=42`, diff after filtering lines matching `BENCH:|Compiled|^I0000|^WARNING`); structure checks (`--xla_dump_to=DIR`, then count `xla_cpu_small_call`, ` while(`, ` scatter(` in `*after_optimizations*.txt`) are load-independent and always valid.

## Building the corpus (target: 25-40 modules, instruction counts spanning ~50 to ~30k)

Non-huge means: compiles in under ~5 s and runs in under ~1 s per call on this machine. Sources, in order of value:

1. **Real regression workloads** (`/Users/xitrium/claud/perf-regression/`): `jaxley-dump/module_0272.jit_run.before_optimizations.txt` (the jax#26145 win), `dumps/frag_0100/frag_0100/module_0021...before_optimizations.txt` (straight-line dense math, historically flat), `mm-dump/` (DUS-heavy, large f64, expect gates inert), `diffrax-dump/module_0014...` (FFI boundaries, expect little effect), `mjx-dump/` (unrolled kinematics).
2. **The discussion-24501 repro:** `/private/tmp/...scratchpad/d24501/dump/module_0000.jit_compute.before_optimizations.txt` if it still exists; otherwise regenerate: jit of `lax.scan(lambda c,i: (c + jnp.sum(jnp.square(jnp.arange(1000000))) + i,)*2, 0, jnp.arange(1000000))[0]` under `venv/bin/python` with `--xla_dump_to`.
3. **Synthetic families:** `reroll-spike/gen_hlo.py` on the [notes branch](https://github.com/seantalts/xla/tree/notes/cpu-small-model-regression-findings/reroll-spike) generates rolled/unrolled/batched families at N in {8,32,128,512}; these calibrate the cap and the chain blowup directly.
4. **The CPU HLO benchmark suite (primary "benchy" middle, internal):** `//platforms/xla/benchmarks/cpu/hlo`. Not visible in the open-source tree; if you are running with internal access, use it as the main corpus and adapt build/run to the internal tooling. IMPORTANT CAVEAT for everything you conclude from it: many of these are partial-model HLO slices, not whole programs. Per-module properties (instruction count, total compile time) measured on them do not represent production modules; per-run and per-region properties do. Weight your analysis accordingly. If running externally instead, substitute the HLO templates embedded in `xla/backends/cpu/benchmarks/*.cc` (elementwise, dot, reductions, gather/scatter, convs, topk): extract 10-15 representative instantiations at the sizes the benchmark files themselves use.
5. **Tiny HLO files:** `xla/tools/data/*.hlo` and `xla/hlo/testdata/*.hlo` for the low end (sanity that the pass is inert or harmless on trivial modules).
6. **Big-end anchors:** Gemma3 1B (`https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo`, 2.3 MB text, needs ~4 GB RAM for fake args) and, if reachable from the registry, `gemma2_2b_keras_jax`. These are deliberately over the "non-huge" line; include them only as the module-gate's far anchors.
7. **The constructed counterexample** for question 1c: append the 24501 scan (as a called computation feeding one output) to a gemma-scale module, or simplest, concatenate its entry into a synthetic module with ~10k instructions of medium dot/elementwise chains. It must be a single module whose runtime is loop-dominated but whose instruction count is big.

For every module record: total post-opt instruction count (`grep -c ' = ' after_opt.txt`), thunk-level instruction count (instructions in the entry plus non-fusion called computations; a ~20-line python script over the dump), and whether it contains while loops / scatters.

## Protocol

Phase A (corpus baseline, no code changes): for each module, interleaved D/S/D/S at `--iters` sized so a run takes 2-10 s; record compile ms, run ms, parity, region count, and classify runtime as win (>5%), wash, or loss, same for compile. Deliver the table sorted by instruction count, and separately flag which corpus entries are partial-model slices (their module-level numbers carry the caveat from question 1). The question-1 deliverable is not a module threshold: it is whether any module loses at runtime under the local gates, and what the compile-cost curve looks like on the big end, counterexample included.

Phase B (cap, code change): in `xla/service/cpu/small_region_hoisting_pass.cc`, phase 2 of `PartitionComputation`: route a run through segmentation when `run.total_bytes >= small_buffer_access_size || run.members.size() >= cap`, and add `segment.members.size() >= cap` as a flush trigger next to the byte check. The existing keep-rule (amplified or contains-control-flow) automatically turns entry-chain caps into declines. Plumb `cap` as a constructor arg with a default; strict RED-GREEN: write tests first (an entry chain of `cap+k` small ops must not fold; a while-body chain of `cap+k` ops must split into multiple calls; jaxley-shaped bodies below the cap unchanged). Then measure: unrolled_{32,128,512} compile (expect return to sentinel-level), jaxley runtime at cap in {32, 64, 96, 128} (its ~80-member regions split at the low caps; quantify), d24501 and gemma unchanged. Recommend a value.

Phase C (question 4, code change behind a bool): apply the keep-rule to unsegmented runs as well, i.e. a region folds only if amplified or contains control flow. Re-run the whole corpus. If nothing regresses beyond noise and the straight-line winners are confirmed absent, recommend adopting it and deleting the entry-side cap semantics from Phase B (the amplified-side cap stays).

## Deliverables

1. A results file (markdown, one table per phase) committed to the [notes branch](https://github.com/seantalts/xla/tree/notes/cpu-small-model-regression-findings) under `gating-experiments/`, including every command and the corpus manifest with file hashes or generation commands.
2. Recommended gate set with the evidence: whether local gates suffice (expected) or what additional compile-cost control is needed on big modules; cap value; question-4 verdict.
3. If code changes are kept: tests green (`//xla/service/cpu:small_region_hoisting_pass_test` and `small_scatter_expander_test`), jaxley and 24501 parity re-verified, on a branch named `feat/region-gating-experiments` pushed to `seantalts/xla`.

## Rules

- Strict RED-GREEN for any pass change; never weaken an existing test to make it pass, flag semantic conflicts instead.
- Bitwise parity accompanies every runtime claim. A result without parity is not a result.
- Commit identity: `git -c user.name="seantalts" -c user.email="talts@google.com"`; never add AI attribution of any kind to commits, code, or docs.
- Prose for anything public: no em-dashes anywhere, and match the plain, intensifier-free register of the existing docs on this branch.
- If a measurement contradicts this brief's expectations, believe the measurement, say so loudly, and re-derive; this project has overturned its own assumptions four times on exactly that discipline.
