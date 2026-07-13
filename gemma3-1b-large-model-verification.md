# Agent G: Gemma3 1B flax_call vs SmallRegionHoisting + SmallScatterExpander

Date: 2026-07-13. Host: Mac Studio (Apple Silicon), binary `/Users/xitrium/claud/xla-defrag/bazel-bin/xla/tools/bench_hlo` (prebuilt, branch under test).
Sentinel = `XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0"` (both passes off, approximates upstream).

## Artifact

- URL: https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo
- Size: **2,338,863 bytes (2.3 MB)** — not hundreds of MB. MD5 `f0d28366b373969c519c6d8d110a30ff` (matches GCS etag).
- Module: `HloModule jit___call__` — a single Gemma3-1B forward call, **batch=1, seq=11** (tokens `s32[1,11]`), all weights bf16. Prefill-style; no decode while-loop in the input.
- ENTRY: `%main.10141 (Arg_0.1: bf16[262144,1152], Arg_1.2: bf16[1152], ... Arg_366.367: s32[1,11])` — embedding table bf16[262144,1152], per-layer QKV/MLP weights, final arg is the token ids.
- ROOT: tuple of logits `bf16[1,262144]` + per-layer KV-cache tensors `bf16[1,11,1,256]` + `s32[1]` cache indices.
- Parameters: **367**, total **1.863 GB** (2,000,064,916 bytes) of bf16 weights → runtime footprint ~2-3 GB with fake args; no memory issues.

### Op census (input HLO)

| op | count |
|---|---|
| ` while(` | 0 |
| ` scatter(` | 22 |
| ` dynamic-update-slice(` | 52 |
| ` gather(` | 2 |
| ` dot(` | 183 |
| instruction lines (`^\s+\S+ = `) | 9,335 |
| total lines / computations | 10,949 / 403 |

All 22 scatters are identical tiny index scatters: `s32[11] scatter(s32[11], s32[11,1], s32[11])`, one per attention layer (metadata `.../layer_N/attn/cond/branch_0_fun/scatter`). Footprint ~132 B — far below the 64 KB gate, so SmallScatterExpander is expected to fire on ALL of them.

### Registry: other CPU-relevant model benchmarks (default_registry.yml)

- `gemma2_2b_keras_jax` — GCS `xla-benchmarking-temp/gemma2_2b_keras_jax.hlo`, CPU_X86 config, CPU_TIME metric.
- `gemma4_2b_bf16` — in-tree `xla/tools/benchmarks/hlo/hlo_gemma4_2b_bf16.hlo`, CPU_X86 config.
- `nv_maxtext_1n1g_jit_train_step_before_optimization.hlo` — GPU_B200 only, not CPU-relevant.

## Results

| config | compile ms (rep1 / rep2) | mean run ms (rep1 / rep2, n=20 w=3) | median ms |
|---|---|---|---|
| DEFAULT (passes on) | 812.97 / 813.24 | 164.12 / 166.85 | 164.12 / 166.13 |
| SENTINEL (passes off) | 722.84 / 713.11 | 171.10 / 167.98 | 172.28 / 167.59 |

(Initial default probe at --iters=5 --warmup=1: compile 912.1 ms, mean 166.70 — per-iter < 200 ms so scaled to 20/3 per protocol.)

- **Runtime**: default 164.1–166.8 vs sentinel 168.0–171.1. Ranges nearly touch; call it a **wash with a slight (~1-3%) lean toward default**. Not a regression.
- **Compile time**: default consistently **~95-100 ms slower (+13%)** (813 vs 718 ms) — the passes plus 26 extra call regions and 22 scatter→loop expansions cost real compile work, though absolute cost is small.

## Parity

`--print_result --seed=42`, one run per config, output filtered with `grep -vE "BENCH:|Compiled|^I0000|^WARNING"` then `shasum -a 256`:

- default:  `83bf74b6de333ea66a7eed724a37a6c886c48d02d5f6a2921392cca6d6ac1a5e`
- sentinel: `83bf74b6de333ea66a7eed724a37a6c886c48d02d5f6a2921392cca6d6ac1a5e`

**Bit-identical. PASS.**

## Structure (default config, after_optimizations dump in `dumpG/`)

| metric | count |
|---|---|
| `xla_cpu_small_call` occurrences | **26** |
| ` scatter(` surviving | **0** (22 in input → all expanded) |
| ` while(` | **22** (all carry `.../attn/.../scatter` metadata → SmallScatterExpander loops) |
| ` dynamic-update-slice(` | 22 |
| ` gather(` | 1 |

Small-call regions: 26 total — 22 regions of 19 instructions (one per layer; op mix per region: 9 fusions, 1 conditional, params/tuples; e.g. `%convert_bitcast_fusion.153_region`, result `(f32[44,11], bf16[1,1,11,1,256], f32[11,256])` — per-token attention-mask/rope/KV-slot bookkeeping) and 4 regions of 11 instructions.

## Interpretation

The passes are **not inert** on this model: even a 1B-parameter graph carries per-layer small-tensor bookkeeping (KV-slot index scatters, mask/rope index math at seq=11), and both passes fired exactly where predicted — 22/22 scatters expanded into while+DUS loops and 26 small-call regions hoisted, one per attention layer. Despite firing on every layer, runtime is a wash to slightly favorable (~1-3% lean toward default, within noise overlap), because the model is dominated by the 183 bf16 dots that the passes don't touch. Numerics are bit-identical. The only cost is ~13% (+95 ms) compile time on an 800 ms compile — worth noting but not a shipping blocker at this scale; worth re-checking on a larger module (gemma2_2b / gemma4_2b in the same registry) where region count scales with layers. No red flags for default-on: byte gates behaved exactly as designed, firing only on genuinely tiny per-token structures without perturbing the compute-bound path.

## Command log

```
curl -sI https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo
curl -sS -o gemma3_1b_flax_call.hlo https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo
md5 gemma3_1b_flax_call.hlo
grep -c ' while(' / ' scatter(' / ' dynamic-update-slice(' / ' gather(' / ' dot(' gemma3_1b_flax_call.hlo ; wc -l
python3 <parse ENTRY params, sum bytes>   # 367 params, 1.863 GB
bench_hlo --hlo_file=gemma3_1b_flax_call.hlo --iters=5 --warmup=1                     # default probe
bench_hlo --hlo_file=gemma3_1b_flax_call.hlo --iters=20 --warmup=3                    # default rep1, rep2
XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0" \
  bench_hlo --hlo_file=gemma3_1b_flax_call.hlo --iters=20 --warmup=3                  # sentinel rep1, rep2
bench_hlo ... --iters=1 --warmup=0 --print_result --seed=42 | grep -vE "BENCH:|Compiled|^I0000|^WARNING" | shasum -a 256   # both configs
XLA_FLAGS="--xla_dump_to=<scratchpad>/dumpG" bench_hlo ... --iters=1 --warmup=0       # default structure dump
grep -c 'xla_cpu_small_call' / ' scatter(' / ' while(' dumpG/module_0001.jit___call__.cpu_after_optimizations.txt
python3 <region size/op-mix analysis>
```
