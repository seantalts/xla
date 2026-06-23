# Stage-1 Region Compilation (XLA:CPU) — Feasibility Scoping & Investigation Record

**Author:** seantalts · **Date:** 2026-06-23 · **Status:** scoping complete, GO

## Purpose

After the Stage-0 partial fix for jaxley (#26145) landed (`c710a3a39d`: body-internal region hoisting via the legacy `ComputationKernelEmitter`, 1.33×, recovering ~38% of the regression), the remaining ~3× requires collapsing *across* scatter boundaries — i.e. putting scatter **inside** a region kernel. The legacy substrate can't (`IrEmitter::HandleScatter` is `Unimplemented`). "Stage-1" keeps the hoisting pass + `xla_cpu_small_call` attribute but swaps the executor to the **MLIR/xtile** stack, which is believed to emit scatter. This doc records whether that's true, what building it requires, the realistic payoff, the increment plan, and **exactly how each load-bearing question was investigated** (so the conclusions are reproducible, not asserted).

This is scoping only — no production code was changed during this investigation.

## The four load-bearing questions

1. Is scatter actually MLIR-emittable on CPU today? (Gates everything — if no, Stage-1-via-MLIR doesn't unblock jaxley.)
2. Can the MLIR stack emit a whole multi-op region as one kernel, or is it strictly one-fusion-per-kernel (region driver = greenfield)?
3. What's the realistic ceiling — does collapsing across scatters approach the old 373µs?
4. What's the minimal, risk-sequenced first increment?

---

## Q1 — Scatter CPU emission path: **MLIR-emittable = YES** (verified)

**Finding.** Scatter on CPU is emitted by a dedicated MLIR fusion emitter, `CpuScatterFusion` (a `KernelEmitter<MlirKernelSource>`), not the legacy elemental emitter and not a `ScatterThunk`. The legacy `IrEmitter::HandleScatter` is dead for the fusion path.

**How this was investigated (steps, reproducible):**
- Grepped the emitter trees for scatter handling → `xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h:41`: `class CpuScatterFusion final : public KernelEmitter<MlirKernelSource>`.
- Found the dispatch site: `xla/service/cpu/thunk_emitter.cc:824-825` —
  `… fusion->fused_expression_root()->opcode() == HloOpcode::kScatter) { auto kernel_emitter = std::make_unique<CpuScatterFusion>(…)`. So a scatter-rooted fusion is routed to the MLIR emitter and compiled via the fusion compiler into a plain `KernelThunk`.
- Confirmed the legacy path is dead: `xla/service/cpu/ir_emitter.cc:1896` → `HandleScatter` returns `Unimplemented("Scatter is not implemented on CPUs.")`.
- **Empirical proof (the decisive step):** dumped the jaxley module's kernels and inspected scatter-kernel provenance.
  `XLA_FLAGS="--xla_dump_to=/tmp/sc" bench_hlo --hlo_file=…/jaxley-dump/module_0272.jit_run.before_optimizations.txt --iters=1`
  → `module_0001.jit_run.wrapped_scatter.1.ir-no-opt.ll` contains the provenance tag
  `xla_cpu_emitter__cpu_scatter_fusion__hlo_opcode__fusion`.
  That tag is emitted by `CpuScatterFusion`, proving scatter goes through the MLIR stack in production right now (jaxley has dozens of such `wrapped_scatter` kernels).
- The jaxley scatters are tiny (`f32[32]` operand, scalar update, `update_window_dims={}` — dump lines ~1601-1974), i.e. dispatch-bound, not compute-bound — exactly what collapsing into a region removes.

**Conclusion:** the premise holds. Scatter can live inside an MLIR region kernel.

---

## Q2 — Multi-op region emission: **one-fusion-per-kernel today; a region driver is greenfield** (but sits on existing machinery)

**Finding.** Every CPU MLIR emission entry point is keyed on a single `HloFusionInstruction` with one hero root. There is no entry point that emits a general `HloComputation` (a region) by walking its schedule. So Stage-1 needs a new **region driver**; but the per-op emission, partitioning, scatter lowering, and kernel ABI all already exist and are reusable.

**How this was investigated:**
- `xla/backends/cpu/codegen/fusion_emitter.cc:284` — `EmitFusionKernel(…, const HloFusionInstruction& fusion, …)` switches on `fusion_kind()` / `FindNonTrivialHero(...)`; handles only loop/concatenate/DUS.
- `CpuScatterFusion` ctor takes `const HloFusionInstruction*`; `EmitKernelDefinition` emits exactly one entry func for one fusion (`cpu_scatter_emitter.cc:256-331`).
- Grepped both emitter trees for `EmitComputation|RegionKernel|EmitRegion` → no general-computation entry point.
- Reusable substrate found: `xla/codegen/emitters/computation_partitioner.h:151-155` — `class PartitionedComputations` / `explicit PartitionedComputations(const HloComputation* …)`. Despite the name it partitions **any** `HloComputation` into subgraphs and provides a `CallTargetProvider` mapping subgraphs → `func.func`s. A region's outlined computation IS an `HloComputation`, so this + `SubgraphToMlirFunction` + `elemental_hlo_to_mlir.*` is exactly the substrate a driver orchestrates.
- Structural template confirmed: `xla/backends/cpu/codegen/computation_kernel_emitter.h:47` — `class ComputationKernelEmitter final : public KernelEmitter<LlvmKernelSource>`; its `EmitNestedComputation` (`.cc:235`) delegates the whole `to_apply()` to the legacy `IrEmitter`. The Stage-1 emitter is the `MlirKernelSource` analog.

**What to build:** `RegionKernelEmitter : public KernelEmitter<MlirKernelSource>` in `xla/backends/cpu/codegen/emitters/`. Its `EmitKernelDefinition` builds an MLIR entry func (port the KernelSpec/buffer-slice plumbing from `CpuScatterFusion::EmitKernelDefinition` + `EmitEntryFunctionApi`, `cpu_fusion_emitter.cc:175`), then runs a **region driver** that walks the outlined computation's schedule and per op: (a) emits fusions as region-local `func.func`s via `PartitionedComputations`/`SubgraphToMlirFunction`; (b) inlines small dots; (c) lowers `kWhile`/`kConditional` to `scf.while`/`scf.if`; (d) materializes non-escaping intermediates as tensor/alloca rather than buffer slices (escaping values keep slices). Route it via the same `xla_cpu_small_call` attribute in `thunk_emitter.cc`, gated behind `xla_cpu_experimental_region_compilation`.

**The genuinely new code is the driver** (schedule walk + control-flow structuralization + intermediate localization). Everything it calls exists.

---

## Q3 — Realistic ceiling

- Per-thunk floor is ~0.5–1.2µs (design doc §1.1); jaxley pays it on dozens of scatter dispatches plus surrounding fusion/dot dispatches; current bench ~1.12ms, old non-thunk ~373µs.
- The jaxley scatters are `f32[32]`/scalar-update → intrinsic scatter compute is trivial (`scf.if` + one store per index); almost none of the residual ~3× is irreducible scatter work — it's dispatch + buffer-table traffic + inter-scatter intermediate round-trips, which is exactly what a single region kernel with allocas removes.
- **Estimate: 373µs is roughly the floor and is largely reachable.** Expect Stage-1 to land in the high-100s to low-400s µs — closing the bulk of the remaining 3×, not a fraction. Caveat: v1 regions are single-workgroup (design §2.2) — fine here, the work is small.

---

## Q4 — Risk-sequenced increment plan

**Increment 1 — straight-line `RegionKernelEmitter`, no scatter, no control flow, behind `xla_cpu_experimental_region_compilation`.** New `region_kernel_emitter.{h,cc}`; driver walks a straight-line outlined computation, emits each op as a region-local function, threads tensor SSA, keeps intermediates in-function. Gate at the `ComputationKernelEmitter` selection site in `thunk_emitter.cc` (flag off ⇒ Stage-0 legacy path unchanged). Validate: bit-parity with Stage-0 on frag + synthetic straight-line; bench ≥ Stage-0 (alloca win should already show on multi-intermediate regions).

**Increment 2 — scatter inside the driver (the increment that unblocks jaxley's 3×).** Reuse `EmitScatterComputation` + the `scf.if` bounds-check from `cpu_scatter_emitter.cc:151,420-465`, invoked when the driver hits a `kScatter` in schedule order (in-place on the region tensor). Loosen `InstructionIsUnavailable` to allow scatter inside regions when the flag is on. Validate on the jaxley module vs interpreter.

**Increment 3 — control flow (`scf.while`/`scf.if`).** Highest risk (loop-carried values, in-place across iterations); deferred last. Validate on fori_loop / slow-MWE.

Rationale: one capability per increment (matches design §3 "each stage changes one thing"); scatter (the win) sits behind a proven straight-line substrate so a scatter-lowering bug can't be confused with a driver bug.

---

## Go/No-Go

**GO.** The single load-bearing risk (scatter MLIR-emittability) is retired with empirical proof. Remaining work is the region driver — greenfield, but on top of fully-existing partitioning / elemental / scatter / kernel-ABI machinery, with `ComputationKernelEmitter` as the structural template.

**Alternative if the premise had been false (it wasn't):** (a) implement `HandleScatter` in the legacy `IrEmitter` to keep the `ComputationKernelEmitter` substrate, or (b) dispatch-fast-path-only (design §4) — bounded at ~2-3× and explicitly insufficient.

## Investigation method note

Conclusions here were reached by: grepping the emitter trees for dispatch/handlers; reading the cited `file:line` sites; and — for the load-bearing scatter question — an *empirical* dump (`--xla_dump_to`) of the real jaxley module confirming the `cpu_scatter_fusion` kernel-provenance tag, not just static code reading. The four key claims (CpuScatterFusion exists + is routed; legacy HandleScatter dead; PartitionedComputations takes a general HloComputation; ComputationKernelEmitter is the template) were independently re-verified after the initial spike.

## Key files

`xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.{h:41,cc:151,256,420}` · `cpu_fusion_emitter.cc:175` · `xla/backends/cpu/codegen/fusion_emitter.cc:284` · `xla/service/cpu/thunk_emitter.cc:824` · `xla/backends/cpu/codegen/computation_kernel_emitter.{h:47,cc:235}` · `xla/codegen/emitters/computation_partitioner.h:151` · `xla/service/cpu/ir_emitter.cc:1896` · design `/Users/xitrium/claud/cpu-region-compilation-design.md:51-86`.
