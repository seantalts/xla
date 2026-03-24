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

## Design

A `CpuOptimizationPreset` enum in `DebugOptions` (`xla.proto` field 502):

| Value | Meaning |
|-------|---------|
| `CPU_OPTIMIZATION_PRESET_DEFAULT` (0) | Maps to `RUNTIME_PERFORMANCE` |
| `CPU_OPTIMIZATION_PRESET_FAST_COMPILE` (1) | Fast compilation + strict numerics |
| `CPU_OPTIMIZATION_PRESET_RUNTIME_PERFORMANCE` (2) | Full optimization for max runtime speed |

The preset is a **compound flag** — it sets defaults for many individual flags. Explicit user overrides via individual flags or `xla_backend_extra_options` still take precedence.

## Phase 0: Flag Defaults Per Preset

This phase surveys every flag that affects CPU compilation and defines what each preset sets by default. No new passes — just changing flag defaults.

### Flag Survey

#### HLO Pass Pipeline Flags (controlled in `cpu_compiler.cc`)

| Flag / Behavior | `RUNTIME_PERFORMANCE` (current default) | `FAST_COMPILE` | Where controlled |
|----------------|----------------------------------------|----------------|------------------|
| `CpuInstructionFusion` pass | **Enabled** | **Skipped** | `cpu_compiler.cc:1034` |
| Tiled emitter (`EnableTiledEmitter`) | **Enabled** (unless `xla_cpu_disable_tiled_emitter` set) | **Disabled** | `cpu_compiler.cc:1049-1050` |
| `CpuMultiOutputFusion` pass | **Enabled** (if `xla_cpu_use_multi_output_fusion` set) | **Skipped** (fast_compile forces off) | `cpu_compiler.cc:1027` |
| `ParallelTaskAssigner` | **Enabled** | **Skipped** | `cpu_compiler.cc:1098` |
| `FlattenAfterFusion` (inline fusions) | Per `xla_cpu_flatten_after_fusion` extra option | Per `xla_cpu_flatten_after_fusion` extra option (no change) | `cpu_compiler.cc:954,1060` |

#### LLVM Compilation Flags (controlled in `IrCompiler::Options`)

These are set in `cpu_compiler.cc` lines ~2108-2124 (JIT) and ~2252-2268 (AOT):

| Flag | `RUNTIME_PERFORMANCE` | `FAST_COMPILE` | Source |
|------|----------------------|----------------|--------|
| `optimization_level` (LLVM CodeGen) | From `xla_backend_optimization_level` (default 3 → `CodeGenOptLevel::Aggressive`) | **Forced to O1** (`CodeGenOptLevel::Less`), overrides global flag | `cpu_compiler.cc:2111` |
| `optimize_for_size` | `false` (unless `xla_cpu_optimize_for_size` extra option set) | **`true`** | `cpu_compiler.cc:2111` |
| `disable_expensive_passes` | `false` (unless `xla_llvm_disable_expensive_passes`) | **`true`** | `cpu_compiler.cc:2116` |
| `slp_vectorizer_disabled` | `false` (unless `xla_cpu_disable_slp_vectorizer` extra option) | **`true`** | `cpu_compiler.cc:2118` |
| `disable_loop_unrolling` | `false` (unless `xla_cpu_disable_loop_unrolling` extra option) | **`true`** | `cpu_compiler.cc:2120` |
| `disable_platform_dependent_math` | `false` (unless `xla_cpu_disable_platform_dependent_math` extra option) | **`true`** | `cpu_compiler.cc:2122` |
| `fast_math_flags` | From `GetCpuFastMathFlags()` — depends on `xla_cpu_enable_fast_math` (default `false`, so **no fast-math by default**) | From `GetCpuFastMathFlags()` — **no change needed** (already off by default) | `cpu_compiler.cc:2115` |

#### Numerics / Fast-Math Flags (controlled via `DebugOptions`)

| Flag | Default | `RUNTIME_PERFORMANCE` | `FAST_COMPILE` | Notes |
|------|---------|----------------------|----------------|-------|
| `xla_cpu_enable_fast_math` | `false` | `false` — **consider `true`** | `false` (strict numerics) | Gates all LLVM fast-math flags. Currently off for both — runtime_perf could benefit from enabling. |
| `xla_cpu_fast_math_honor_nans` | `true` | `true` | `true` | Only relevant if `enable_fast_math=true` |
| `xla_cpu_fast_math_honor_infs` | `true` | `true` | `true` | Only relevant if `enable_fast_math=true` |
| `xla_cpu_fast_math_honor_division` | `true` | `true` | `true` | Only relevant if `enable_fast_math=true` |
| `xla_cpu_fast_math_honor_functions` | `true` | `true` | `true` | Only relevant if `enable_fast_math=true` |
| `xla_cpu_enable_fast_min_max` | `true` | `true` | `true` — **consider `false`** | Disables NaN propagation in min/max. Affects `AlgebraicSimplifier`. |
| `xla_cpu_enable_platform_dependent_math` | `true` | `true` | `true` — **consider `false`** | Allows platform-specific math implementations |

#### Other CPU Flags (no change per preset)

| Flag | Default | Notes |
|------|---------|-------|
| `xla_backend_optimization_level` | `3` | Generic LLVM CodeGen opt level. Preset overrides this for CPU via `IrCompiler::Options.optimization_level`. |
| `xla_cpu_use_fusion_emitters` | `true` | Both presets use MLIR fusion emitters. |
| `xla_cpu_prefer_vector_width` | `256` | Hardware-dependent, not preset-dependent. |
| `xla_cpu_max_isa` | Platform default | Hardware-dependent, not preset-dependent. |
| `xla_cpu_parallel_codegen_split_count` | `32` | LLVM module splitting for parallel compile — keep for both (reduces LLVM wall time). |
| `xla_cpu_scheduler_type` | `DEFAULT` | Orthogonal scheduling concern. |
| `xla_cpu_use_onednn` | `false` | Library dispatch, not preset-dependent. |
| `xla_cpu_use_xnnpack` | `true` | Library dispatch, not preset-dependent. |
| `xla_cpu_use_acl` | `true` | Library dispatch, not preset-dependent. |
| `xla_cpu_multi_thread_eigen` | `true` | Runtime threading, not compile-time. |
| `xla_cpu_strict_dot_conv_math` | unset | Precision control, orthogonal. |
| `xla_llvm_enable_alias_scope_metadata` | `true` | Keep for both (cheap, helps LLVM). |
| `xla_llvm_enable_noalias_metadata` | `true` | Keep for both. |
| `xla_llvm_enable_invariant_load_metadata` | `true` | Keep for both. |
| `xla_llvm_force_inline_before_split` | `true` | Keep for both (aids parallel compile). |

#### Backend Extra Options (via `xla_backend_extra_options`)

These are the granular overrides in `cpu_options.h`. The preset sets defaults; users can still override:

| Extra Option | `RUNTIME_PERFORMANCE` default | `FAST_COMPILE` default | Notes |
|-------------|------------------------------|----------------------|-------|
| `xla_cpu_optimize_for_size` | not set | **force on** | Already handled via `is_fast_compile` |
| `xla_cpu_disable_slp_vectorizer` | not set | **force on** | Already handled |
| `xla_cpu_disable_loop_unrolling` | not set | **force on** | Already handled |
| `xla_cpu_disable_platform_dependent_math` | not set | **force on** | Already handled |
| `xla_cpu_disable_new_fusion_emitters` | not set | not set | Both use fusion emitters |
| `xla_cpu_disable_tiled_emitter` | not set | **force on** | Already handled via `is_fast_compile` |
| `xla_cpu_flatten_after_fusion` | not set | not set | No change — fusion flattening is orthogonal |
| `xla_cpu_use_multi_output_fusion` | not set | not set | Skipped in fast_compile via pass gating |
| `xla_cpu_fold_all_constants` | not set | not set | Orthogonal |

### What `is_fast_compile` Currently Controls (Code Audit)

From `cpu_compiler.cc`, when `IsFastCompileMode()` returns true:

1. **Line 1034**: Skip `CpuInstructionFusion` pass
2. **Line 1041-1044**: Run `MegaFusionPass` instead (Phase 1 — see Future Work)
3. **Line 1050**: Disable tiled emitter
4. **Line 1027**: Disable multi-output fusion
5. **Line 1098**: Skip `ParallelTaskAssigner`
6. **Lines 2111-2123**: Force LLVM options (CodeGen O1, optimize_for_size, disable_expensive_passes, disable_slp_vectorizer, disable_loop_unrolling, disable_platform_dependent_math)

### Summary: Phase 0 Changes for `FAST_COMPILE`

| Category | Change | Impact |
|----------|--------|--------|
| HLO fusion | Skip `CpuInstructionFusion` | Fewer, simpler fusions → less LLVM work |
| HLO fusion | Skip `CpuMultiOutputFusion` | Less fusion overhead |
| Tiled emitter | Disabled | Simpler MLIR lowering |
| Parallel tasks | Skip `ParallelTaskAssigner` | Less analysis time |
| LLVM | CodeGen optimization level forced to **O1** (`CodeGenOptLevel::Less`) | Faster instruction selection and register allocation |
| LLVM | `optimize_for_size = true` | Smaller code, faster LLVM passes |
| LLVM | `disable_expensive_passes = true` | Skip costly LLVM analyses |
| LLVM | `disable_slp_vectorizer = true` | Skip SLP vectorization |
| LLVM | `disable_loop_unrolling = true` | Skip loop unrolling |
| LLVM | `disable_platform_dependent_math = true` | Use portable libm, not inline approx |

**No change** to `RUNTIME_PERFORMANCE` — it is the current default behavior.

### Resolved Design Decisions

1. **LLVM CodeGen O1**: `FAST_COMPILE` forces `llvm::CodeGenOptLevel::Less` (O1), overriding the global `xla_backend_optimization_level` (default 3). This applies to both JIT and AOT paths in `cpu_compiler.cc`.

2. **`xla_cpu_enable_fast_math`**: Left as `false` for both presets. Enabling it for `RUNTIME_PERFORMANCE` would be a semantic change affecting existing users — this can be discussed with affected teams case by case if needed.

3. **`xla_cpu_enable_fast_min_max`**: Left as `true` (default) for both presets. No change needed.

## Usage

```
--xla_cpu_optimization_preset=CPU_OPTIMIZATION_PRESET_FAST_COMPILE
```

The flag is orthogonal to `xla_backend_optimization_level` (which controls only LLVM CodeGen opt level). Explicit user overrides via `xla_backend_extra_options` still take precedence.

## Future Work

### Phase 1: MegaFusion Pass

The current `FAST_COMPILE` preset skips `CpuInstructionFusion` but still relies on `FusionWrapper` to wrap individual ops into single-op fusions. This means each op becomes a separate LLVM function — N ops = N functions = N × LLVM compile cost.

A **MegaFusion pass** would wrap as many ops as possible into a single large fusion (or small number of large fusions), then emit through the MLIR `LoopFusionKernelEmitter`. Benefits:

1. **Single kernel** — one LLVM function to compile, minimal thunk dispatch overhead
2. **Register intermediates** — values between fused ops stay in registers, no buffer round-trips
3. **Fast LLVM compile** — one function instead of N means less LLVM work

**Proposed algorithm:**
1. Iterate each computation in reverse post-order
2. Create seed `kLoop` fusions, greedily absorb single-user fusible operands via `FuseInstruction()`
3. `FusionWrapper` runs afterward to wrap any remaining unfusible single ops

#### Non-Fusible Ops (Thunk Barriers)

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

For a small elementwise model with no dots/convolutions, the result would be literally **one thunk** executing **one LLVM function**.

### Phase 2: Additional Presets

As needs arise, additional presets could be added (e.g. a `DEBUG` preset for near-zero compile time, or presets that independently control numerics vs compile speed).
