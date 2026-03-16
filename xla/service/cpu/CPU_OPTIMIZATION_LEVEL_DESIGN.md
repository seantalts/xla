# XLA:CPU Optimization Level Flag (`xla_cpu_optimization_level`)

## Motivation

XLA:CPU compiles HLO through a pipeline of fusion passes, MLIR lowering, and LLVM optimization. This pipeline is tuned for peak runtime performance but pays a fixed compilation cost. For small models (e.g. tiny inference graphs, unit-test-sized computations), this cost dominates — users wait seconds to compile something that runs in microseconds.

We need a way to say "just emit this simply" without regressing the architecture toward the legacy `elemental_ir_emitter` LLVM path. Both optimization levels must use MLIR lowering exclusively.

## Architecture Context: Emitter Paths

Understanding the current emitter landscape is critical to this design:

| Path | Entry Point | Lowering | Status |
|------|-------------|----------|--------|
| **MLIR fusion emitters** | `FusionWrapper` → `EmitFusionKernel` → `LoopFusionKernelEmitter` | HLO fusion → MLIR → LLVM IR | **Forward-looking path** |
| **MLIR tiled emitters** | `FusionWrapper` → `EmitFusionKernel` → `TiledFusionEmitter` | HLO fusion → MLIR (tiled) → LLVM IR | Forward-looking (O2) |
| **Elemental LLVM emitter** | `EmitElementalKernelThunk` → `ElementalKernelEmitter` | HLO op → `CpuElementalIrEmitter` → LLVM IR | **Legacy path** (despite the name, this uses `CpuElementalIrEmitter`) |
| **IrEmitter2 fusion fallback** | `EmitFusionHostKernel` → `IrEmitter::HandleFusion` | HLO fusion → `ElementalIrEmitter` → LLVM IR | **Legacy path** |

**Key insight:** `ElementalKernelEmitter` (used for individual unfused ops) actually uses `CpuElementalIrEmitter` under the hood — it emits LLVM IR directly, not MLIR. The only true MLIR path is through the **fusion emitters** (`LoopFusionKernelEmitter`, `TiledFusionEmitter`, etc.). Therefore, to stay on the MLIR path, **all ops must go through fusions**.

## Design

A new `CpuOptimizationLevel` enum in `DebugOptions` with three values:

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

**Fusion strategy:** A greedy mega-fusion pass that:
- Merges all fusible ops (elementwise, broadcast, reshape, reduce, gather, scatter, slice, pad, iota, etc.) into one big `kLoop` fusion
- Leaves non-fusible ops as separate thunks: `kDot` (→ Eigen), `kConvolution` (→ library), `kCustomCall`, `kCollective*`, `kWhile`, `kConditional`, `kFft`, `kSort`, `kInfeed`/`kOutfeed`
- Uses `FusionWrapper` on remaining individual fusible ops that couldn't join the mega-fusion (e.g. due to data dependencies on non-fusible ops creating barriers)

**What the compilation pipeline looks like in O1:**

```
HLO passes (no CpuInstructionFusion) → MegaFusionWrapper → ThunkEmitter
  ├─ mega-fusion → LoopFusionKernelEmitter (MLIR) → FusionCompiler → single LLVM kernel
  ├─ dot → DotThunk (Eigen library call)
  ├─ custom_call → CustomCallThunk
  └─ remaining fusible op → FusionWrapper → LoopFusionKernelEmitter (MLIR)
```

**HLO passes skipped:**
- `CpuInstructionFusion` — replaced by mega-fusion strategy
- `CpuMultiOutputFusion` — not needed
- `ParallelTaskAssigner` — overhead exceeds benefit for small ops

**LLVM options forced on:**
- `optimize_for_size` — smaller code
- `disable_expensive_passes` — skip costly LLVM passes
- `disable_slp_vectorizer` — skip SLP auto-vectorization
- `disable_loop_unrolling` — skip loop unrolling
- `disable_platform_dependent_math` — portable math

### O2: Full Optimization (Default)

Current behavior. `CpuInstructionFusion` → `FusionWrapper` (tiled emitter + loop fusion) → multi-output fusion → parallel task assignment → full LLVM optimization.

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
