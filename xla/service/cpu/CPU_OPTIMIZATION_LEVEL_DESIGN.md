# XLA:CPU Optimization Level Flag (`xla_cpu_optimization_level`)

## Motivation

XLA:CPU compiles HLO through a pipeline of fusion passes, MLIR lowering, and LLVM optimization. This pipeline is tuned for peak runtime performance but pays a fixed compilation cost. For small models (e.g. tiny inference graphs, unit-test-sized computations), this cost dominates — users wait seconds to compile something that runs in microseconds.

We need a way to say "just emit this simply" without regressing the architecture toward the legacy `elemental_ir_emitter` LLVM path. Both optimization levels must use MLIR lowering exclusively.

### Prior Art in XLA

GPU has granular compile-time knobs (`xla_gpu_autotune_level` 0–4, `xla_gpu_disable_gpuasm_optimizations`, `xla_gpu_cudnn_gemm_fusion_level`) rather than a single optimization tier — autotuning is GPU's dominant compile-time cost. TPU uses the global `EffortLevel` in `ExecutionOptions` (O0–O3). The generic `xla_backend_optimization_level` controls only the LLVM CodeGen opt level across backends. CPU is the first backend with a **unified O1/O2 mode** that jointly controls HLO passes, fusion strategy, emitter selection, and LLVM options.

## Architecture Context: Emitter Paths

| Path | Entry Point | Lowering | Status |
|------|-------------|----------|--------|
| **MLIR fusion emitters** | `FusionWrapper` → `EmitFusionKernel` → `LoopFusionKernelEmitter` | HLO fusion → MLIR → LLVM IR | **Forward-looking path** |
| **MLIR tiled emitters** | `FusionWrapper` → `EmitFusionKernel` → `TiledFusionEmitter` | HLO fusion → MLIR (tiled) → LLVM IR | Forward-looking (O2) |
| **Elemental LLVM emitter** | `EmitElementalKernelThunk` → `ElementalKernelEmitter` | HLO op → `CpuElementalIrEmitter` → LLVM IR | **Legacy path** |
| **IrEmitter2 fusion fallback** | `EmitFusionHostKernel` → `IrEmitter::HandleFusion` | HLO fusion → `ElementalIrEmitter` → LLVM IR | **Legacy path** |

**Key insight:** `ElementalKernelEmitter` uses `CpuElementalIrEmitter` under the hood — LLVM IR, not MLIR. The only true MLIR path is through the **fusion emitters**. Therefore, to stay on the MLIR path, **all ops must go through fusions**.

## Design

A `CpuOptimizationLevel` enum in `DebugOptions` (`xla.proto` field 502):

| Value | Meaning |
|-------|---------|
| `CPU_OPTIMIZATION_LEVEL_DEFAULT` (0) | Maps to O2 |
| `CPU_OPTIMIZATION_LEVEL_O1` (1) | Fast compilation |
| `CPU_OPTIMIZATION_LEVEL_O2` (2) | Full optimization (current behavior) |

### O1: Fast Compilation — Mega-Fusion Strategy

The core idea: wrap as many ops as possible into a **single large fusion** (or a small number of large fusions), then emit each through the MLIR `LoopFusionKernelEmitter`. This gives us:

1. **MLIR lowering** — stays on the forward-looking path, no `CpuElementalIrEmitter`
2. **Single kernel** — one LLVM function to compile, minimal thunk dispatch overhead
3. **Register intermediates** — values between fused ops stay in registers, no buffer round-trips
4. **Fast LLVM compile** — one function instead of N functions means less LLVM work

### MegaFusionPass Implementation

**Files:** `mega_fusion_pass.{h,cc}` — an `HloModulePass` that runs in O1 mode before `FusionWrapper`.

**Algorithm:**
1. For each (non-fusion) computation, iterate in reverse post-order (consumers before producers)
2. For each unprocessed fusible instruction, create a seed `kLoop` fusion via `HloInstruction::CreateFusion()`
3. Repeatedly scan the fusion's operands: if an operand (a) passes `CanFuse()` and (b) has exactly **one user** (this fusion), absorb it via `fusion->FuseInstruction(operand)`
4. Repeat step 3 until no more operands qualify, then move to the next unprocessed instruction
5. `FusionWrapper` runs afterward to wrap any remaining single fusible ops

The **single-user constraint** is critical — it prevents duplicating an instruction's computation across multiple fusions. If an op feeds two consumers, it stays outside both fusions as a shared operand.

**Fusible ops** (mirrors `CanBeLoopFused` + the MLIR elemental emitter's supported set): all elementwise ops, plus bitcast, broadcast, concatenate, dynamic_slice, dynamic_update_slice, gather, iota, pad, reduce, reduce_window, reshape, reverse, scatter, slice, transpose. **Not fusible:** dot, convolution, custom_call, collectives, control flow, fft, sort, tuple/GTE, parameter, constant, existing fusions.

**Constraints:** No nested fusions (MLIR emitter rejects `kFusion` inside a fusion). If the fusion root is scatter, `ThunkEmitter` routes through `CpuScatterFusion` instead of `LoopFusionKernelEmitter`. Dot as root falls back to the legacy emitter. Both `xla_cpu_use_fusion_emitters` and `UseExperimentalLoopFusion` flags must be enabled.

### O1 Pipeline (`cpu_compiler.cc`)

```
HLO passes (no CpuInstructionFusion) → MegaFusionPass → FusionWrapper → ThunkEmitter
  ├─ mega-fusion → LoopFusionKernelEmitter (MLIR) → FusionCompiler → single LLVM kernel
  ├─ dot → DotThunk (Eigen library call)
  ├─ custom_call → CustomCallThunk
  └─ remaining fusible op → FusionWrapper → LoopFusionKernelEmitter (MLIR)
```

**HLO passes skipped:** `CpuInstructionFusion`, `CpuMultiOutputFusion`, `ParallelTaskAssigner`.

**LLVM options forced on:** `optimize_for_size`, `disable_expensive_passes`, `disable_slp_vectorizer`, `disable_loop_unrolling`, `disable_platform_dependent_math`.

### O2: Full Optimization (Default)

Current behavior. `CpuInstructionFusion` (pairwise producer-consumer fusion with profitability heuristics) → `FusionWrapper` (tiled emitter + loop fusion) → multi-output fusion → parallel task assignment → full LLVM optimization.

## Non-Fusible Ops (Thunk Barriers)

These ops cannot be absorbed into a mega-fusion and create thunk boundaries:

| Op | Reason | Thunk type |
|----|--------|------------|
| `kDot` | Eigen/oneDNN library dispatch | DotThunk |
| `kConvolution` | Library dispatch | ConvolutionThunk |
| `kCustomCall` | Arbitrary external code | CustomCallThunk |
| `kFft` | Library dispatch | FftThunk |
| `kSort` | Complex comparator logic | SortThunk |
| `kWhile` / `kConditional` | Host-side control flow | WhileThunk / ConditionalThunk |
| `kAllReduce` / `kAllGather` / ... | Cross-device coordination | CollectiveThunks |
| `kInfeed` / `kOutfeed` | Device I/O | I/O Thunks |

For a small elementwise model with no dots/convolutions, the result is literally **one thunk** executing **one LLVM function**.

## Usage

```
--xla_cpu_optimization_level=CPU_OPTIMIZATION_LEVEL_O1
```

The flag is orthogonal to `xla_backend_optimization_level` (which controls only LLVM CodeGen opt level). Explicit user overrides via `xla_backend_extra_options` still take precedence.

## Future Work

- **O0**: Skip even more (algebraic simplification fixed-point, CSE). Near-zero compile time for debugging.
- **O3**: Speculative optimizations not yet default (aggressive constant folding, experimental fusion strategies).
- As MLIR emitter coverage grows (e.g. dot, conv get MLIR emitters), the mega-fusion can absorb more ops.
