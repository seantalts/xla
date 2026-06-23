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

## UPDATE 2026-06-23 — Increment 1 attempt hit a substrate wall; increment plan revised

Built the Increment-1 scaffold (TDD): `RegionKernelEmitter : KernelEmitter<MlirKernelSource>` (`xla/backends/cpu/codegen/emitters/region_kernel_emitter.{h,cc}`), flag `xla_cpu_experimental_region_compilation` (via `xla_backend_extra_options`), routed in `thunk_emitter.cc` at the `ComputationKernelEmitter` selection site, with `IsSupportedRegion` gating + clean fallback to the legacy emitter. All staged (not committed), tests pass, and **parity verified** (flag on/off bitwise-identical on frag + a synthetic straight-line region) — because the emitter currently *declines* everything real and falls back. The build is a safe scaffold, not a working MLIR region emitter yet.

**Wall (verified):** `xla/codegen/emitters/computation_partitioner.cc:497` builds the tensor-parameter + per-output index-arg ABI **only** for `computation->IsFusionComputation() || IsEntryComputation()`. An `xla_cpu_small_call` region is a plain `kCall` `to_apply()` (neither), so `CreateSubgraphMlirFunction` emits index-less scalar reducer-style funcs while the subgraph's `index_ranges` expect index args → `SubgraphToMlirFunction`'s `drop_front(num_parameters)` underflows (`SmallVector ... capacity 18446744073709551615`). Emitting a region as one MLIR kernel therefore requires the region to *be* a fusion computation, or broadening this ABI gate.

**Second finding (re-scopes the increment order):** pure straight-line *single-shape* elementwise chains never reach the hoisting pass as `kCall`s — ordinary loop *instruction fusion* already collapses them into a single `kLoop` fusion (verified: synthetic chains emerge as one fusion, not a small_call). The `xla_cpu_small_call` regions that actually exist are **multi-shape** (frag: `dot→tanh→reduce→divide→broadcast→subtract` over `f32[16,16]`/`f32[16]`/`f32[]`; `call.99` returns a 4-tuple of differing shapes). So "straight-line single-shape region via MLIR" both (a) hits the wall and (b) duplicates loop fusion and isn't exercised by real workloads.

**Why this is bigger than an Increment-1 wrapper:** the MLIR substrate's model is "one subgraph func computes one output element of the root from the *computation parameters*" (`ProvideParameter`, `elemental_hlo_to_mlir.cc:1286`, always re-derives operands via `take_front(num_parameters)`). There is no mechanism to thread an intermediate SSA *tensor* between differently-shaped subgraphs. A multi-shape region needs a genuine region driver: per-op loop nests in schedule order, intermediates materialized as tensors/allocas and threaded.

**Revised increment plan (supersedes Q4 above):**
- **Increment 1' (the real substrate, two candidate approaches — DECISION NEEDED):**
  - **A. Region-as-fusion.** Have `SmallRegionHoistingPass` emit the region as a `kFusion` (multi-output) instead of/in addition to the `kCall`+attr, so the existing fusion ABI + partitioner apply unchanged. Pro: reuses all fusion machinery. Con: fusion semantics (single hero root, elementwise-rooted) may not cleanly model a multi-shape region containing dots/reduces; design §4 already flags that fusion can't model multi-output-with-control-flow (control flow is Increment 3, so straight-line multi-shape may be OK).
  - **B. Generalize the partitioner + multi-shape driver.** Broaden `CreateSubgraphMlirFunction`'s ABI gate to general region computations and build the schedule-walk driver that materializes/threads intermediate tensors. Pro: the genuine, general region substrate the design anticipated. Con: more new code; touches shared partitioner infra (cross-backend blast radius).
- **Increment 2 — scatter inside the driver** (unblocks jaxley 3×). Unchanged from original plan; depends on 1'.
- **Increment 3 — control flow.** Unchanged.

## Investigation method note

Conclusions here were reached by: grepping the emitter trees for dispatch/handlers; reading the cited `file:line` sites; and — for the load-bearing scatter question — an *empirical* dump (`--xla_dump_to`) of the real jaxley module confirming the `cpu_scatter_fusion` kernel-provenance tag, not just static code reading. The four key claims (CpuScatterFusion exists + is routed; legacy HandleScatter dead; PartitionedComputations takes a general HloComputation; ComputationKernelEmitter is the template) were independently re-verified after the initial spike.

## Key files

`xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.{h:41,cc:151,256,420}` · `cpu_fusion_emitter.cc:175` · `xla/backends/cpu/codegen/fusion_emitter.cc:284` · `xla/service/cpu/thunk_emitter.cc:824` · `xla/backends/cpu/codegen/computation_kernel_emitter.{h:47,cc:235}` · `xla/codegen/emitters/computation_partitioner.h:151` · `xla/service/cpu/ir_emitter.cc:1896` · design `/Users/xitrium/claud/cpu-region-compilation-design.md:51-86`.
