// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-lower-tensors="target_type=cpu" \
// RUN: | FileCheck %s

func.func @load_non_gep_from_args(%arg0: !llvm.ptr) -> !llvm.ptr {
  %0 = llvm.getelementptr inbounds %arg0[1]
    : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
  %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
  func.return %2 : !llvm.ptr
}

// CHECK-LABEL: @load_non_gep_from_args
// CHECK-NEXT:    %0 = llvm.getelementptr inbounds %arg0[1]
// CHECK-NEXT:    %1 = llvm.load %0 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %2 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    return %2 : !llvm.ptr

// -----

func.func @slice_copy_to_memcpy(%src: tensor<512xf32> {xla.slice_index = 0},
    %dst: tensor<768xf32> {xla.slice_index = 1}, %i: index, %j: index)
    -> tensor<768xf32> {
  %run = tensor.extract_slice %src[%i] [128] [1]
    : tensor<512xf32> to tensor<128xf32>
  %out = tensor.insert_slice %run into %dst[%j] [128] [1]
    : tensor<128xf32> into tensor<768xf32>
  func.return %out : tensor<768xf32>
}

// CHECK-LABEL: @slice_copy_to_memcpy(
// CHECK-SAME:    %[[SRC:.*]]: !llvm.ptr {xla.slice_index = 0 : i64},
// CHECK-SAME:    %[[DST:.*]]: !llvm.ptr {xla.slice_index = 1 : i64},
// CHECK-SAME:    %[[I:.*]]: index, %[[J:.*]]: index) {
// CHECK-DAG:     %[[SIZE:.*]] = llvm.mlir.constant(512 : i64) : i64
// CHECK-DAG:     %[[J_I64:.*]] = arith.index_castui %[[J]] : index to i64
// CHECK-DAG:     %[[DST_PTR:.*]] = llvm.getelementptr inbounds %[[DST]][0, %[[J_I64]]]
// CHECK-SAME:      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<768 x f32>
// CHECK-DAG:     %[[I_I64:.*]] = arith.index_castui %[[I]] : index to i64
// CHECK-DAG:     %[[SRC_PTR:.*]] = llvm.getelementptr inbounds %[[SRC]][0, %[[I_I64]]]
// CHECK-SAME:      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<512 x f32>
// CHECK:         "llvm.intr.memcpy"(%[[DST_PTR]], %[[SRC_PTR]], %[[SIZE]]) <{isVolatile = false}>
// CHECK-NOT:     tensor.
// CHECK:         return

// -----

func.func @slice_copy_in_loop(%src: tensor<64xf32> {xla.slice_index = 0},
    %dst: tensor<128xf32> {xla.slice_index = 1}) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %dst)
      -> (tensor<128xf32>) {
    %src_offset = arith.muli %i, %c16 : index
    %dst_offset = arith.addi %src_offset, %c16 : index
    %run = tensor.extract_slice %src[%src_offset] [16] [1]
      : tensor<64xf32> to tensor<16xf32>
    %next = tensor.insert_slice %run into %acc[%dst_offset] [16] [1]
      : tensor<16xf32> into tensor<128xf32>
    scf.yield %next : tensor<128xf32>
  }
  func.return %out : tensor<128xf32>
}

// CHECK-LABEL: @slice_copy_in_loop(
// CHECK-SAME:    %[[SRC:.*]]: !llvm.ptr {xla.slice_index = 0 : i64},
// CHECK-SAME:    %[[DST:.*]]: !llvm.ptr {xla.slice_index = 1 : i64}) {
// CHECK:         scf.for
// CHECK-NOT:       iter_args
// CHECK:           llvm.getelementptr inbounds %[[DST]]
// CHECK:           llvm.getelementptr inbounds %[[SRC]]
// CHECK:           "llvm.intr.memcpy"
// CHECK-NOT:     tensor.

// -----

func.func @whole_tensor_copy_to_memcpy(
    %src: tensor<64xf32> {xla.slice_index = 0 : i64},
    %dst: tensor<128xf32> {xla.slice_index = 1 : i64}) -> tensor<128xf32> {
  %c64 = arith.constant 64 : index
  %out = tensor.insert_slice %src into %dst[%c64] [64] [1]
    : tensor<64xf32> into tensor<128xf32>
  func.return %out : tensor<128xf32>
}

// CHECK-LABEL: @whole_tensor_copy_to_memcpy(
// CHECK-SAME:    %[[SRC:.*]]: !llvm.ptr {xla.slice_index = 0 : i64},
// CHECK-SAME:    %[[DST:.*]]: !llvm.ptr {xla.slice_index = 1 : i64}) {
// CHECK-DAG:     %[[SIZE:.*]] = llvm.mlir.constant(256 : i64) : i64
// CHECK-DAG:     %[[DST_PTR:.*]] = llvm.getelementptr inbounds %[[DST]][0, 64]
// CHECK-DAG:     %[[SRC_PTR:.*]] = llvm.getelementptr inbounds %[[SRC]][0, 0]
// CHECK:         "llvm.intr.memcpy"(%[[DST_PTR]], %[[SRC_PTR]], %[[SIZE]])
