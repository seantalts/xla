# Agent B: LLVM IR quality audit of XLA:CPU region kernels (jaxley module_0272)

Question: within a single-function region kernel emitted by the legacy ComputationKernelEmitter
(SmallRegionHoistingPass outputs, call.93..call.110), does LLVM already (a) forward
intermediate buffer round-trips to registers and (b) vectorize the loops? I.e. how much
headroom does a register-SSA/vectorizing emitter have?

## Commands run

```
XLA_FLAGS="--xla_dump_to=$SCRATCH/dumpB" \
  $SCRATCH/bench_hlo --hlo_file=/Users/xitrium/claud/perf-regression/jaxley-dump/module_0272.jit_run.before_optimizations.txt --iters=1 --warmup=0
# -> compiled in 254 ms; dump dir has 361 files: *.ir-no-opt.ll + *.ir-with-opt.ll per kernel module,
#    module_0001.jit_run.cpu_after_optimizations.txt, buffer assignment, obj files.
grep -n 'call\.93\b|call\.97\b|call\.103\b' dumpB/module_0001.jit_run.cpu_after_optimizations.txt
sed -n '2073,2160p' dumpB/...cpu_after_optimizations.txt   # add_bitcast_fusion.3_region (call.93)
sed -n '2793,2890p' dumpB/...cpu_after_optimizations.txt   # reverse_bitcast_fusion.9_region (call.97)
# store/load/vector/shuffle counts via grep -c; per-buffer round-trip analysis via python re script
# (tracks getelementptr base-pointer chains, counts loads occurring after first store to same buffer).
```

All IR files: `$SCRATCH/dumpB/module_0001.jit_run.call.NN_computation_kernel_module.ir-{no-opt,with-opt}.ll`.

## Region-to-computation mapping (after-opt HLO, line numbers in cpu_after_optimizations.txt)

| call | region computation | contents | shapes / trip counts |
|---|---|---|---|
| call.93 (hottest, 1.297ms/3000) | `%add_bitcast_fusion.3_region` (l.2073) | iota, identity-gather, scalar add, big elementwise fusion (3 exp chains) | f32[32], f32[32,1] → trips 32 |
| call.97 (1.240ms/3000) | `%reverse_bitcast_fusion.9_region` (l.2793) | ~20 kLoop fusions: slices, divides, 4x DUS, 2x DUS+gather, 3x reverse, pad, broadcast-mul, selects, reduce — a vmapped 4x4 tridiagonal (Thomas) solve | f32[1], f32[4], f32[4,4], f32[4,5], f32[16,1], f32[32] → trips 4/16/32 |
| call.103 (1.164ms/3000) | `%reverse_bitcast_fusion.5_region` (l.4097) | same solve pattern at width 1: f32[1], f32[1,4], f32[3,1], f32[4,1] | trips 1–4 |

Kernel function = one LLVM function per region, e.g. `@call.97_kernel(ptr %0)` in
`call.97_computation_kernel_module.ir-with-opt.ll`. Module metadata confirms emitter:
`!xla_cpu_memory_region_name = {"xla_cpu_emitter__computation_kernel_emitter__hlo_opcode__call", "ir_emitter"}`.

## 3. Representative kernel: `@call.97_kernel`

### Headline counts (static instructions)

| metric | no-opt | with-opt |
|---|---|---|
| lines | 2126 | 837 |
| stores | 223 | 75 |
| loads | 381 | 140 (~28 are invariant arg-pointer loads; ~110 data loads) |
| allocas | 47 (all i64 loop induction vars) | 0 |
| memcpy | 2 | 2 |
| vector-typed float ops | 0 | 128 |
| shufflevector | 0 | 46 |
| conditional branches | many | 11 |

### (a) Intermediate materialization — forwarding largely did NOT fire across fusion boundaries

Per-buffer analysis of with-opt IR: **90 of ~110 data loads read a buffer that the same
function stored earlier** — i.e. surviving intermediate round-trips. Worst offenders:
arg87 (13 stores / 25 reloads), arg81 (1 vector store loop / 21 scalar reloads),
arg82 (7/12), arg71 (8/11), arg78 (4/4), arg72 (1/3).

Concrete failures (with-opt, `@call.97_kernel`):

- Immediate scalar reload, 6 lines apart:
  ```llvm
  store float %div.643.i, ptr %arg73, align 64, !alias.scope !25, !noalias !26   ; L69
  tail call void @llvm.memcpy...(%arg94 <- %arg79), !noalias !46                 ; L73
  %7 = load float, ptr %arg73, align 64, !alias.scope !25, !noalias !26          ; L75  <- NOT forwarded
  ```
  Cause: the memcpy's `!noalias !46 = {!16}` names only the buffer-table scope, not arg73's
  buffer scope (!15), so MemDep can't prove no-clobber. Same pattern arg80: store L72, reload L85.
- Vector/scalar shape mismatch defeats GVN: L160 stores `<16 x float> %interleaved.vec376`
  to arg87; L180–211 store 12 floats scalar to strided offsets of arg87; L212–269 then
  reload all 16 floats of arg87 *one scalar at a time* for four row-sum reductions —
  values that were in registers 30 lines earlier.
- The identity-gather in call.93 (`iota_gather_fusion.24`) was strength-reduced to a plain
  vectorized copy loop arg63→arg59 (`vector.body147`), but the *next* loop
  (`vector.body152`) reloads arg59 from memory: LLVM does not fuse adjacent loops, so
  every producer→consumer edge between fusions is a mandatory memory round-trip even
  when both sides vectorize perfectly.

So: the optimizer removed ~2/3 of the naive traffic (223→75 stores, 381→140 loads,
all 47 allocas gone) *within* fusion bodies, but the cross-fusion buffer round-trips —
the thing a register-SSA emitter would eliminate — mostly survive.

### (b) Vectorization — mixed

- Plain elementwise segments: **yes, well vectorized.** call.93's exp-heavy fusion is a clean
  `<4 x float>` loop (trips 8) with the exp polynomial fully inlined, no scalar residue.
- Strided/transposed segments: vectorized but drowning in shuffles. call.97 has 46
  shufflevectors — `<16 x float>` wide loads deinterleaved into 4 strided lanes,
  recombined, re-interleaved (e.g. `vector.body353`, `vector.body370`, `vector.body395`).
  This is data movement that exists only because each fusion writes row-major buffers
  that the next fusion reads in a different order.
- DUS segments: **scalar** with per-row control flow — `dynamic-update-slice.486` is an
  unrolled-x3 scalar body behind `br i1 %147` (freeze/trunc of a pred), 4 outer iterations.
- Reverse segment: **scalar** — `reverse_bitcast_fusion.13` is a 16-iteration scalar loop with
  index arithmetic `xor %288, 3` (the reverse), 2-level loop-unswitched branches, and a
  per-element select; one float store per iteration.
- Slice/divide with predicates: scalar unrolled (`slice_intersection-after47.i[.us].3`),
  unswitched into us/non-us clones.
- call.103 (3rd hottest): **zero vector ops, zero shuffles** in with-opt IR — 56 stores /
  78 loads (26 = round-trips) of pure scalar chains, because all its shapes are 1–4 elements.

### (c) Loop structure

A *sequence* of ~15 independent loop nests / unrolled straight-line blocks, one per inner
fusion, strictly ordered by the memory dependences through buffer slices (all sub-slices of
one arena, buffer index 190). No fused loop nest anywhere. 2 memcpys (copy ops). 0 allocas.
Epilogue builds the result tuple: 19 pointer stores (call.97) / 7 (call.93) into the tuple
buffer per invocation — calling-convention overhead a body-emitter can't remove.

## 4. Reference: standalone small fusion `@gather_divide_fusion` (loop_fusion_kernel_emitter)

`gather_divide_fusion_kernel_module.ir-with-opt.ll`: f32[4,1] output = -(gather(s32 idx) * x) / gather(y).
Fully unrolled to 4 scalar iterations: load i32 index → clamp (`llvm.smax`/`llvm.umin`) →
2 dependent gather loads → fneg/fmul/fdiv → store. No intermediates, no round-trips (single
fusion), scalar because dynamic-index gather can't vectorize on NEON (no HW gather) and
trips=4. IR quality here is essentially optimal for the op; per-invocation cost is dominated
by thunk dispatch, not the ~20 flops. This is the ceiling the new emitter path already hits
for single fusions.

## 5. Verdict

**Headroom of a register-SSA/tiled emitter over the legacy region kernels: MEDIUM on kernel
bodies, modest end-to-end.**

Evidence for headroom (against "LLVM already fixed it"):
- Store-to-load forwarding demonstrably fails across fusion boundaries: 90 of ~110 data
  loads in the optimized `@call.97_kernel` re-read buffers written earlier in the same
  function (26 of 54 in call.103). Causes are structural — incomplete noalias scopes on
  memcpys, vector-store/scalar-reload shape mismatches, and LLVM's inability to fuse
  adjacent loops — so better metadata alone won't fix it; SSA-by-construction would.
- 46 shufflevectors in call.97 are pure layout tax between fusions; a tiled emitter that
  keeps the 4x4 solve in ~8 NEON registers eliminates them and the ~75 stores/90 reloads.
- Reverse and DUS segments are scalar with per-element branches; in register SSA a reverse
  is a shuffle and a DUS with constant offset is an insertelement — both free-ish.
- The whole live set of a region is a few hundred bytes; everything fits in registers.

Evidence limiting headroom (against "new emitter wins big"):
- The compute that dominates flop count (call.93's triple-exp elementwise loop over 32
  elements) is *already* cleanly vectorized; a new emitter gains ~nothing there, and
  call.93 is the hottest region.
- Trip counts are 1–32. Dynamic-index gathers stay scalar under any emitter on NEON.
- Fixed per-invocation overhead is untouched: loading ~28 buffer pointers from the args
  table, 19 tuple-pointer stores, plus KernelThunk dispatch. At ~0.35–0.43 µs per region
  call (profile: 1.0–1.3 ms / 3000 calls) versus an estimated ~0.1 µs of irreducible body
  work, dispatch+ABI plausibly eats a third to a half of each call already.
- LLVM already removed ~2/3 of naive memory traffic; the remaining round-trips are L1-hot
  loads (~4–5 cycle latency each, partly overlapped), not misses.

Quantified guess: eliminating the surviving round-trips + shuffle tax could plausibly
cut region-kernel *body* time by ~1.5–2.5x (call.97/call.103-shaped kernels; ~1x for
call.93-shaped ones). Region calls (call.82–110) sum to roughly a third of while-body
time in the profile (~19 ms visible in top-30 of ~77 ms while.270, plus the tail).
Net end-to-end from IR quality alone: **~10–15% of step time**, unless the new emitter
also reduces per-call dispatch/tuple-ABI overhead or enables larger regions — those are
where the rest of the per-call 0.4 µs lives.
