# Compile-time blowup discrimination (hoisted 128-op unrolled chain)

Question: why does folding a 128-fusion unrolled chain into one region
kernel inflate compile time ~11x (193ms -> 2.17s)?

Method: flag ablation on the same unrolled.hlo (N=128 trace-unrolled
stepping loop), bench_hlo, Apple M3.

| config | compile ms |
|---|---|
| upstream (no fold), split_count=32 (default) | 254 |
| upstream (no fold), split_count=1 | 236 |
| hoisted (one function), split_count=32 | 2184 |
| hoisted (one function), split_count=1 | 2184 |
| hoisted + --xla_llvm_disable_expensive_passes | 2173 |
| hoisted + --xla_backend_optimization_level=1 | 1546 |
| hoisted + --xla_backend_optimization_level=0 | 73 (runtime 30us vs 27us at O3) |

Conclusion: parallel-codegen loss explains nothing (split count is a
no-op in both configs). The entire blowup is LLVM middle-end
optimization of one giant straight-line function (huge basic block,
~1300 memory ops, long live ranges); it persists at O1 and vanishes at
O0 while keeping ~90% of the folding runtime win. Mitigations adopted
in the design doc: member-count cap for control-flow-free regions
(primary), reduced opt level for oversized region functions
(secondary option).
