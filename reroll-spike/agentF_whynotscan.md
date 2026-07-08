# Agent F: Why do JAX libraries unroll loops at trace time, and what does it cost?

Date: 2026-07-08. Environments: `/Users/xitrium/claud/perf-regression/venv` (jax 0.10.1),
`venv-old` (jax 0.4.30). Bench binary: `/Users/xitrium/claud/xla-defrag/bazel-bin/xla/tools/bench_hlo`.
Benchmark script: `loop_tradeoff.py` in this scratchpad. Dumps under `dumpF/` in this scratchpad.

## 1. Why unrolled

### 1a. MJX: unrolled by construction, forced by static heterogeneity

The per-joint/per-body "loops" in MJX kinematics/CRB/RNE are NOT `lax.scan`; they are
Python loops at trace time inside `mujoco/mjx/_src/scan.py` (in venv:
`/Users/xitrium/claud/perf-regression/venv/lib/python3.11/site-packages/mujoco/mjx/_src/scan.py`).

- `scan.body_tree` (scan.py:337-497) — docstring: *"Scan `f` across bodies in tree order,
  carrying results up/down the tree. This function groups bodies according to level and
  attached joints, then calls vmap(f) on them."* The implementation builds a static grouping
  key per body (scan.py:375-405, comment: *"1) the tree depth: parent bodies are processed
  first, so that they are available as carry input to child bodies ... 2) the types of
  arguments passed to f ... for 'j' arguments, we group by joint type; for 'q' arguments, we
  group by q width; for 'v' arguments, we group by dof width"*), then runs a **Python for
  loop over the sorted group keys** (scan.py:449-477, `for key in keys: ... key_y[key] =
  _nvmap(f, carry, *f_args)`). Each iteration is a separate traced `vmap(f)` call — one
  emitted subgraph per (tree-depth, joint-type-signature) group. For the mock-mass-matrix
  35-joint chain every body is its own depth level, so this emits ~1 group per joint →
  the ~208 dynamic-update-slice fusions we measured in jax#26021.
- `scan.flat` (scan.py:168-334) does the same for body/actuator scans: Python loop over
  `key_typ_ids` groups calling `_nvmap(f, ...)` (scan.py:292-297).
- Why it can't be `lax.scan` as written: the scanned callback consumes **static** joint
  metadata and dispatches with Python control flow. `smooth.py:47-94` (`kinematics.fn`) does
  `if jnt_typ == JointType.FREE: ... elif JointType.BALL: ... elif HINGE ... elif SLIDE`,
  and consumes **different qpos widths per joint type** (7/4/1; see `_q_bodyid`,
  scan.py:68-74). `_nvmap` (scan.py:97-130) requires numpy args within a group to be
  identical ("numpy arg elements do not match" RuntimeError) precisely because those values
  are Python-static. `lax.scan` requires a fixed-shape carry and uniformly stacked `xs`
  (jax docstring, below); heterogeneous per-level group sizes and per-joint q/dof widths
  violate the stacked-`xs` constraint, and the tree carry (parent→child) means iteration i's
  input is a gather of earlier outputs, not a homogeneous carry. Rolling it would require
  padding every joint to width 7 + `lax.switch` over joint type, i.e., a rewrite.
- There is **no scan/unroll config option** in `body_tree`/`flat` — no `unroll=` parameter,
  no lax.scan mode. It is unrolled by construction.
- It is not an anti-scan ideology: MJX uses `lax.scan`/`while_loop` where shapes ARE
  homogeneous: `solver.py:253` (`lax.scan(_fun, init, None, length=max_iter)`),
  `solver.py:602` (`lax.while_loop`), `forward.py:402` (RK integrator,
  `jax.lax.scan(f, ..., unroll=3)` — evidence they consciously tune the unroll knob),
  `collision_sdf.py:178`, `support.py:888-889` (TODO comparing scan vs while_loop).
- Perf-tuning intent is visible but about gather/slice, not loops: scan.py:32
  `# TODO(erikfrey): re-check if this really helps perf` on `_take`, whose docstring says
  *"XLA executes x[jp.array([1, 2, 3])] slower than x[1:4], so we detect when take indices
  are contiguous, and convert them to slices."*

Verdict for MJX: **forced by heterogeneity** of the model (joint types/widths per tree
level), not a deliberate "unrolled is faster on CPU" choice. The grouping+vmap design
minimizes the number of unrolled program copies (one per group, not per body), which is
tacit acknowledgment that trace-time unrolling is a cost to be contained.

### 1b. JAX's own stated tradeoff

`jax/_src/lax/control_flow/loops.py` (venv copy), `lax.scan` docstring:

> "scan is a JAX primitive and is lowered to a single WhileOp. That makes it useful for
> reducing compilation times for JIT-compiled functions, since native Python loop
> constructs in a jit function are unrolled, leading to large XLA computations."

> "the loop-carried value `carry` must hold a fixed shape and dtype across all iterations"

`unroll` parameter: "how many scan iterations to unroll within a single iteration of a
loop... unroll=0 unrolls the entire loop... unroll=True completely unrolled". So JAX frames
unrolling purely as a compile-time-vs-(presumed)-runtime knob and exposes a dial; the
structural constraints (fixed-shape carry, uniformly stacked `xs` pytrees) are what make
heterogeneous trees like MJX's hard to roll. `fori_loop` docstring: reduces to `scan` when
the trip count is static (enabling reverse-mode AD), else `while_loop`.

### 1c. diffrax / jaxley

- diffrax 0.7.2 (`diffrax/_integrate.py:480,685` + `_adjoint.py:284-296,390`): rolled —
  equinox `while_loop` with `kind="lax"` / `"checkpointed"` (bounded/checkpointed while
  loops for adjoint support); never trace-time unrolled.
- jaxley 0.5.0 (`jaxley/integrate.py:306` → `jaxley/utils/jax_utils.py:17-74`): rolled —
  `nested_checkpoint_scan`, a recursive-gradient-checkpointing wrapper that "reduces to
  lax.scan"; the 300 time steps are one `lax.scan`.

Both time-steppers roll because their step is homogeneous — same state pytree every
iteration. Unrolling appears only where the iteration space is structurally heterogeneous
(MJX trees) or where someone opts into `unroll=` for throughput (MJX forward.py:402).

## 2. Measured tradeoff at JAX level

Body per step i: `x = state[4i : 4i+4]` (f32[512] state), `y = tanh(W @ x)` (W f32[4,4]),
write back via dynamic-update-slice; N=128 steps in one jit. Three variants: python for
loop (unrolled at trace, static indices), `lax.scan` (dynamic indices), `lax.fori_loop`.
Compilation cache disabled; backend pre-warmed on an unrelated jit; each variant in its own
process; runtime = best of 100 `block_until_ready` calls on the AOT-compiled executable.
Note: second lower/compile always hit an internal cache (0.00 ms) even with
`jax_enable_compilation_cache=False`, so the fresh first compile (after backend warmup, in
a clean process) is the honest number reported below.

Commands:
```
cd /Users/xitrium/claud/perf-regression
for v in unrolled scan fori; do ./venv/bin/python $SCRATCH/loop_tradeoff.py $v; done
for v in unrolled scan fori; do ./venv-old/bin/python $SCRATCH/loop_tradeoff.py $v; done
```

| variant | jax | trace+lower | compile | run best | run median |
|---|---|---|---|---|---|
| unrolled | 0.10.1 | 50.6 ms | 225.1 ms | 22.2 us | 26.2 us |
| scan     | 0.10.1 |  7.1 ms |  18.2 ms |  4.9 us |  5.0 us |
| fori     | 0.10.1 |  6.9 ms |  13.8 ms |  5.0 us |  5.1 us |
| unrolled | 0.4.30 | 76.4 ms | 533.7 ms |  2.6 us |  2.7 us |
| scan     | 0.4.30 |  9.5 ms |  15.4 ms |  2.7 us |  2.9 us |
| fori     | 0.4.30 |  6.8 ms |  11.3 ms |  2.7 us |  2.8 us |

All checksums identical (28.398033/-37, f32 tolerance). Key readings:

1. **Old XLA (0.4.30): unrolling was runtime-free.** 2.6 us unrolled vs 2.7 us scan. The
   only cost was compile (534 ms vs 15 ms, 35x) and trace (76 ms vs 9 ms). Under that
   contract, MJX-style unrolling was rational: pay once at compile, lose nothing at runtime.
2. **Current XLA (0.10.1): unrolling now LOSES at runtime too.** 22.2 us unrolled vs 4.9 us
   scan — 4.5x slower, the per-fusion thunk dispatch overhead of ~128 tiny kernels (this is
   the same mechanism as the 3.7x MJX mock-mass-matrix regression in jax#26021). Compile is
   still 12-16x worse (225 ms vs 14-18 ms). Unrolled went 2.6 → 22.2 us across jax versions
   (8.5x regression) while rolled went 2.7 → 4.9 us.
3. scan and fori are equivalent here (fori with static bounds lowers to scan).

## 3. Our branch (bench_hlo 2x2)

HLO dumped from venv runs (`--xla_dump_to`, `before_optimizations` of `jit_f_unrolled` /
`jit_f_scan`) → `$SCRATCH/unrolled.hlo` (1315 lines), `$SCRATCH/scan.hlo` (126 lines).
`bench_hlo --iters=200 --warmup=20`; upstream approximation =
`XLA_FLAGS="--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0"`.

| config | HLO | compile | run mean |
|---|---|---|---|
| our branch (default)   | unrolled | **2298.7 ms** (stable: 2272/2285/2281 on re-runs) | 24 us |
| our branch (default)   | scan     | 21.0 ms | 15 us (11 us at iters=50) |
| upstream (threshold=0) | unrolled | 210.8 ms | 34 us |
| upstream (threshold=0) | scan     | 22.8 ms | 19 us |

Verification via `--xla_dump_to` on branch runs:
- scan variant: after_optimizations ENTRY is exactly `param, param, ROOT call(...,
  frontend_attributes={xla_cpu_small_call="true"})` — the whole 128-iteration while loop is
  folded into **one kernel**, one dispatch per call. Rolled-form runtime penalty vs
  upstream: none (15 us vs 19 us, i.e. faster).
- unrolled variant: the branch ALSO folds all ~128 DUS fusions into a single
  `xla_cpu_small_call` region (ENTRY = 2 params + 1 call). That fixes upstream's dispatch
  overhead at runtime (34 → 24 us) but hands LLVM one giant straight-line function:
  **compile blows up 210 ms → 2.28 s (10.9x, stable across 4 runs)**. Caveat for our
  branch: region compilation's compile-time cost is superlinear in unrolled program size;
  an MJX/TORAX-scale unrolled module could get much worse. Worth a size cap or splitting
  heuristic in the region former.
- So on our branch the rolled form is the best cell of the 2x2 on BOTH axes: 21 ms compile
  (109x faster than branch-unrolled, 10x faster than upstream-unrolled) and the fastest
  runtime measured in bench_hlo (11-15 us).

(bench_hlo absolute runtimes carry different call overhead than jax-level timings —
compare within the table, not against §2.)

## 4. Verdict

Libraries unroll today for two distinct reasons, and MJX is the hard one: its per-joint
loops are unrolled **by construction** because the loop body dispatches on static,
heterogeneous metadata (Python `if jnt_typ == FREE/BALL/HINGE/SLIDE` with 7/4/1-wide qpos
slices, smooth.py:59-88) that violates lax.scan's fixed-shape-carry/stacked-xs contract —
there is no unroll knob to turn off, and rolling it would mean a pad-and-switch rewrite.
The softer reason is the old performance contract, which our 0.4.30 measurement confirms:
unrolling used to cost only compile time (534 ms vs 15 ms) and zero runtime, so "unroll and
let XLA fuse" was a rational habit; homogeneous-step libraries (diffrax checkpointed
while_loop, jaxley nested lax.scan) never needed it. That contract is now broken: on jax
0.10.1 the same unrolled program is 4.5x slower at runtime than lax.scan (22 vs 5 us) on
top of 12x compile cost — trace-time unrolling is a pure loss on current XLA:CPU, which is
exactly the MJX jax#26021 regression in miniature. Our branch closes the remaining excuse
for unrolling: it folds the scan's while loop into a single `xla_cpu_small_call` kernel, so
the rolled form compiles in 21 ms and is the fastest variant at runtime — strictly
dominating both unrolled cells. The guidance therefore becomes "roll your loops; the
runtime penalty is gone," with two footnotes: (1) heterogeneous-tree code like MJX cannot
roll without a pad/switch redesign, so our branch's other half (cheap handling of unrolled
small-fusion chains) still matters for them — though today it buys its 34→24 us runtime
win at a 10.9x compile-time cost (2.28 s) that needs a cap before it's shippable at
TORAX scale; and (2) jax's own scan docstring already tells users rolled loops "reduce
compilation times" — what changed is that CPU runtime now agrees.
