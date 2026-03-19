# XLA:CPU Optimization Preset (`xla_cpu_optimization_preset`)

## Motivation

XLA:CPU compiles HLO through a pipeline of fusion passes, MLIR lowering, and LLVM optimization. This pipeline is tuned for peak runtime performance but pays a fixed compilation cost. For small models (e.g. tiny inference graphs, unit-test-sized computations), this cost dominates — users wait seconds to compile something that runs in microseconds.

We observe three user profiles:

1. **Compile-time sensitive** — interactive notebooks, unit tests, model iteration. Want fast compile+run, care about correct numerics.
2. **Numerics sensitive** — scientific computing, verification, reference implementations. Need strict FP semantics, don't care about runtime speed.
3. **Runtime performance sensitive** — serving, training, batch inference. Want maximum throughput/latency, accept relaxed FP semantics.

Profiles (1) and (2) naturally unify: the "straightforward" compilation strategy (no fast-math, no aggressive LLVM passes, simple fusion) gives both fast compilation AND strict numerics. There's no tension — fast-math approximations aren't faster to *compile*, they're faster to *execute*. Disabling them simplifies compilation and preserves accuracy.

### Prior Art in XLA

GPU has granular compile-time knobs (`xla_gpu_autotune_level` 0–4, `xla_gpu_disable_gpuasm_optimizations`, `xla_gpu_cudnn_gemm_fusion_level`) rather than a single optimization preset — autotuning is GPU's dominant compile-time cost. TPU uses the global `EffortLevel` in `ExecutionOptions` (O0–O3). The generic `xla_backend_optimization_level` controls only the LLVM CodeGen opt level across backends. CPU is the first backend with a **unified preset** that jointly controls HLO passes, fusion strategy, emitter selection, LLVM options, and numerics mode.

## Architecture Context: Emitter Paths

| Path | Entry Point | Lowering | Status |
|------|-------------|----------|--------|
| **MLIR fusion emitters** | `FusionWrapper` → `EmitFusionKernel` → `LoopFusionKernelEmitter` | HLO fusion → MLIR → LLVM IR | **Forward-looking path** |
| **MLIR tiled emitters** | `FusionWrapper` → `EmitFusionKernel` → `TiledFusionEmitter` | HLO fusion → MLIR (tiled) → LLVM IR | Forward-looking (runtime_performance) |
| **Elemental LLVM emitter** | `EmitElementalKernelThunk` → `ElementalKernelEmitter` | HLO op → `CpuElementalIrEmitter` → LLVM IR | **Legacy path** |
| **IrEmitter2 fusion fallback** | `EmitFusionHostKernel` → `IrEmitter::HandleFusion` | HLO fusion → `ElementalIrEmitter` → LLVM IR | **Legacy path** |

**Key insight:** `ElementalKernelEmitter` uses `CpuElementalIrEmitter` under the hood — LLVM IR, not MLIR. The only true MLIR path is through the **fusion emitters**. Therefore, to stay on the MLIR path, **all ops must go through fusions**.

## Design

A `CpuOptimizationPreset` enum in `DebugOptions` (`xla.proto` field 502):

| Value | Meaning |
|-------|---------|
| `CPU_OPTIMIZATION_PRESET_DEFAULT` (0) | Maps to `RUNTIME_PERFORMANCE` |
| `CPU_OPTIMIZATION_PRESET_FAST_COMPILE` (1) | Fast compilation + strict numerics |
| `CPU_OPTIMIZATION_PRESET_RUNTIME_PERFORMANCE` (2) | Full optimization for max runtime speed |

### `fast_compile`: Fast Compilation + Strict Numerics

Wraps as many ops as possible into a **single large fusion** (or a small number of large fusions) via `MegaFusionPass`, then emits each through the MLIR `LoopFusionKernelEmitter`. This gives us:

1. **MLIR lowering** — stays on the forward-looking path, no `CpuElementalIrEmitter`
2. **Single kernel** — one LLVM function to compile, minimal thunk dispatch overhead
3. **Register intermediates** — values between fused ops stay in registers, no buffer round-trips
4. **Fast LLVM compile** — one function instead of N functions means less LLVM work
5. **Strict numerics** — no fast-math flags, no FP reassociation, no FMA contraction, libm calls for exp/log/tanh instead of inline approximations

**MegaFusionPass algorithm:**
1. Iterate each computation in reverse post-order (consumers first)
2. Create seed `kLoop` fusions, greedily absorb single-user fusible operands via `FuseInstruction()`
3. `FusionWrapper` runs afterward to wrap any remaining single fusible ops

**HLO passes skipped:** `CpuInstructionFusion`, `CpuMultiOutputFusion`, `ParallelTaskAssigner`.

**LLVM options forced on:** `optimize_for_size`, `disable_expensive_passes`, `disable_slp_vectorizer`, `disable_loop_unrolling`, `disable_platform_dependent_math`.

### `runtime_performance`: Full Optimization (Default)

Current behavior. `CpuInstructionFusion` (pairwise producer-consumer fusion with profitability heuristics) → `FusionWrapper` (tiled emitter + loop fusion) → multi-output fusion → parallel task assignment → full LLVM O2 optimization → fast-math enabled.

| Dimension | `fast_compile` | `runtime_performance` |
|-----------|---------------|----------------------|
| Fusion strategy | Mega-fusion (maximal) | Pairwise heuristic |
| exp/log/tanh | libm call (exact) | Inline approximation (fast) |
| FP reassociation | No | Yes |
| FMA contraction | No | Yes |
| LLVM opt level | O1 | O2 |
| Tiled emitter | No | Yes |
| Parallel tasks | No | Yes |

### Pipeline (`cpu_compiler.cc`)

```
fast_compile:
  HLO passes (no CpuInstructionFusion) → MegaFusionPass → FusionWrapper → ThunkEmitter
    ├─ mega-fusion → LoopFusionKernelEmitter (MLIR) → single LLVM kernel
    ├─ dot → DotThunk (Eigen library call)
    └─ remaining op → FusionWrapper → LoopFusionKernelEmitter (MLIR)

runtime_performance:
  HLO passes → CpuInstructionFusion → FusionWrapper (tiled) → MultiOutputFusion
    → ParallelTaskAssigner → ThunkEmitter → full LLVM O2
```

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

For a small elementwise model with no dots/convolutions, the result is literally **one thunk** executing **one LLVM function**.

## Usage

```
--xla_cpu_optimization_preset=CPU_OPTIMIZATION_PRESET_FAST_COMPILE
```

The flag is orthogonal to `xla_backend_optimization_level` (which controls only LLVM CodeGen opt level). Explicit user overrides via `xla_backend_extra_options` still take precedence.

## Future Work

- As MLIR emitter coverage grows (e.g. dot, conv get MLIR emitters), the mega-fusion can absorb more ops.
- Additional presets if needed (e.g. a future `debug` preset for near-zero compile time).
