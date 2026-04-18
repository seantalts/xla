# Whole-Program Codegen for Small XLA:CPU Programs

Branch: `claude/optimize-thunk-runtime-xWU05`

## 1. Problem & Goals

## 2. Non-Goals (v1 Scope Fence)

## 3. High-Level Architecture

## 4. Component Specs

### 4.1 CPU Cost Model (kflops + parallelism)

### 4.2 MegaFusionPass (HLO pass)

### 4.3 Flag plumbing

### 4.4 Pipeline integration

## 5. Detailed Algorithm: MegaFusionPass

### 5.1 Preconditions & per-computation gate

### 5.2 Schedule-order region growing

### 5.3 Absorption predicate (per-op)

### 5.4 Emitting the kFusion instruction

### 5.5 Tuple/multi-output handling

### 5.6 Buffer aliasing & layout constraints

## 6. Data Structures

## 7. Worked Examples

### 7.1 Example 1: 35-joint mass-matrix (pure forward)

### 7.2 Example 2: Cholesky-like scan (while body)

## 8. File-by-File Change List

## 9. Test Plan

### 9.1 Unit tests

### 9.2 Integration / benchmarks

### 9.3 Regression guards

## 10. Rollout & Flag Semantics

## 11. Risks & Open Questions

## 12. v2 Roadmap

### 12.1 While-loop inlining

### 12.2 Parallelism-aware region splitting

### 12.3 Cost model refinement
