# XLA:CPU Optimization Level Flag (`xla_cpu_optimization_level`)

## Motivation

XLA:CPU compiles HLO through a pipeline of fusion passes, MLIR lowering, and LLVM optimization. This pipeline is tuned for peak runtime performance but pays a fixed compilation cost. For small models (e.g. tiny inference graphs, unit-test-sized computations), this cost dominates — users wait seconds to compile something that runs in microseconds.

We need a way to say "just emit this simply" without regressing the architecture toward the legacy `elemental_ir_emitter` LLVM path. Both optimization levels must use MLIR lowering exclusively.

## Design

A new `CpuOptimizationLevel` enum in `DebugOptions` with three values:

| Value | Meaning |
|-------|---------|
| `CPU_OPTIMIZATION_LEVEL_DEFAULT` (0) | Maps to O2 |
| `CPU_OPTIMIZATION_LEVEL_O1` (1) | Fast compilation |
| `CPU_OPTIMIZATION_LEVEL_O2` (2) | Full optimization (current behavior) |

### O1: Fast Compilation

**HLO passes skipped:**
- `CpuInstructionFusion` — no fusions are created, so every op arrives at the ThunkEmitter individually and is lowered through `ElementalKernelEmitter` (MLIR elemental path). This avoids any fallback to the legacy `elemental_ir_emitter`.
- `FusionWrapper` — no tiled emitter, no loop fusion.
- `CpuMultiOutputFusion` — no multi-output fusion.
- `ParallelTaskAssigner` — no intra-op parallelism (overhead exceeds benefit for small ops).

**LLVM options forced on:**
- `optimize_for_size` — smaller code, less to emit.
- `disable_expensive_passes` — skip costly LLVM optimization passes.
- `disable_slp_vectorizer` — skip SLP auto-vectorization.
- `disable_loop_unrolling` — skip loop unrolling.
- `disable_platform_dependent_math` — use portable math (simpler to compile).

**What O1 keeps:** All correctness-required passes (layout assignment, scatter/gather expansion, copy insertion, algebraic simplification, DCE). Library rewrites (oneDNN, YNNPACK) still run if enabled — they're library calls, not elemental emission. The MLIR lowering path is always used.

### O2: Full Optimization (Default)

Current behavior. Instruction fusion → FusionWrapper (tiled emitter + loop fusion) → multi-output fusion → parallel task assignment → full LLVM optimization. This is the forward-looking path toward complete MLIR-based codegen.

## Key Constraint: No `elemental_ir_emitter`

Both levels lower through MLIR. In O1, the path is:

```
HLO op → ThunkEmitter::EmitElementalKernelThunk → ElementalKernelEmitter (MLIR) → LLVM IR
```

In O2, fused ops additionally go through:

```
HLO fusion → FusionCompiler / ParallelFusionEmitter (MLIR tiled) → LLVM IR
```

The legacy `IrEmitter::EmitFusionHostKernel → ElementalIrEmitter` path is never hit in O1 because no fusions are created (CpuInstructionFusion is skipped).

## Usage

```
--xla_cpu_optimization_level=CPU_OPTIMIZATION_LEVEL_O1
```

The flag is orthogonal to `xla_backend_optimization_level` (which controls only LLVM CodeGen opt level, default 3). Explicit user overrides via `xla_backend_extra_options` (e.g. `xla_cpu_disable_slp_vectorizer`) still take precedence — O1 defaults are additive, not exclusive.

## Future Work

- **O0**: Could skip even more (algebraic simplification fixed-point iterations, CSE). Useful for debugging or when compile time must be near-zero.
- **O3**: Could enable speculative optimizations not yet on by default (aggressive constant folding, experimental fusion strategies).
- As MLIR emitter coverage grows, O1's `ElementalKernelEmitter` path naturally improves without any flag changes.
