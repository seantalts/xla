// RUN: emitters_opt %s -one-shot-bufferize -canonicalize -cse \
// RUN: -split-input-file | FileCheck %s

// CHECK: @extract_strided(%[[SOURCE:.*]]: memref<16xf32>, %[[OFFSET:.*]]: index)
func.func @extract_strided(%source: memref<16xf32>, %tile_id: index) -> tensor<8xf32> {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index

  // CHECK: %[[SHIFT:.*]] = arith.subi %[[C15]], %[[OFFSET]] : index
  // CHECK: %[[STRIDED_SHIFT:.*]] = arith.divsi %[[SHIFT]], %[[C2]] : index
  // CHECK: %[[ELEMENTS_TO_END:.*]] = arith.addi %[[STRIDED_SHIFT]], %[[C1]] : index
  // CHECK: %[[SIZE:.*]] = arith.minsi %[[ELEMENTS_TO_END]], %[[C8]] : index
  // CHECK: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE]], %[[C8]] : index

  // CHECK: %[[BUFFER:.*]] = scf.if %[[IS_FULL_TILE]] -> (memref<8xf32>) {
    // CHECK: %[[STATIC_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [8] [2]
    // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<8xf32>
    // CHECK: memref.copy %[[STATIC_SUBVIEW]], %[[ALLOC_0]]
    // CHECK: scf.yield %[[ALLOC_0]] : memref<8xf32>
  // CHECK: } else {
    // CHECK: %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]]
    // CHECK-SAME: [%[[OFFSET]]] [%[[SIZE]]] [2]
    // CHECK-SAME: : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>

    // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<8xf32>

    // CHECK: %[[ALLOC_1_SUBVIEW:.*]] = memref.subview %[[ALLOC_1]]
    // CHECK-SAME: [0] [%[[SIZE]]] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>

    // CHECK: memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_1_SUBVIEW]]
    // CHECK-SAME: : memref<?xf32, strided<[2], offset: ?>> to memref<?xf32, strided<[1]>>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<8xf32>
  // CHECK: }

  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[BUFFER]]
  // CHECK-SAME: : memref<8xf32> to tensor<8xf32>
  %tile = xtile.extract %source[%tile_id][8][2] : memref<16xf32> -> tensor<8xf32>
  // CHECK: return %[[TILE]] : tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

// CHECK: @insert_strided(
// CHECK-SAME: %[[SOURCE:.*]]: tensor<8xf32>,
// CHECK-SAME: %[[DESTINATION:.*]]: memref<16xf32>,
// CHECK-SAME: %[[OFFSET:.*]]: index)
func.func @insert_strided(%source: tensor<8xf32>, %destination: memref<16xf32>, %tile_id: index) {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index

  // CHECK: %[[SOURCE_BUFFER:.*]] = bufferization.to_buffer %[[SOURCE]]
  // CHECK-SAME: : tensor<8xf32> to memref<8xf32, strided<[?], offset: ?>>

  // CHECK: %[[SHIFT:.*]] = arith.subi %[[C15]], %[[OFFSET]] : index
  // CHECK: %[[STRIDED_SHIFT:.*]] = arith.divsi %[[SHIFT]], %[[C2]] : index
  // CHECK: %[[ELEMENTS_TO_END:.*]] = arith.addi %[[STRIDED_SHIFT]], %[[C1]] : index
  // CHECK: %[[SIZE:.*]] = arith.minsi %[[ELEMENTS_TO_END]], %[[C8]] : index
  // CHECK: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE]], %[[C8]] : index

  // CHECK: scf.if %[[IS_FULL_TILE]] {
    // CHECK:   %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]][%[[OFFSET]]] [8] [2]
    // CHECK:   memref.copy %[[SOURCE_BUFFER]], %[[DESTINATION_SUBVIEW]]
  // CHECK: } else {
    // CHECK: %[[SOURCE_SUBVIEW:.*]] = memref.subview %[[SOURCE_BUFFER]][0] [%[[SIZE]]] [1]
    // CHECK-SAME: : memref<8xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>

    // CHECK: %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]]
    // CHECK-SAME: [%[[OFFSET]]] [%[[SIZE]]] [2]
    // CHECK-SAME: : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>

    // CHECK: memref.copy %[[SOURCE_SUBVIEW]], %[[DESTINATION_SUBVIEW]]
    // CHECK-SAME: : memref<?xf32, strided<[?], offset: ?>>
    // CHECK-SAME: to memref<?xf32, strided<[2], offset: ?>>
  // CHECK: }

  xtile.insert %source into %destination[%tile_id][8][2] : tensor<8xf32> -> memref<16xf32>
  return
}

// -----

// Test: identity-layout extract feeding a layout-altering op (stablehlo.transpose)
// should force a local buffer allocation instead of yielding the subview directly.

// CHECK-LABEL: @extract_identity_transpose
func.func @extract_identity_transpose(%source: memref<64xf32>, %tile_id: index) -> tensor<8xf32> {
  // The tile is 8 x f32 = 32 bytes (below register threshold), unit stride,
  // so identity layout. But the user is stablehlo.transpose → force alloc.

  // CHECK: scf.if
  // Full-tile branch: should alloc + copy due to transpose user.
  // CHECK: memref.subview
  // CHECK: memref.alloc() : memref<8xf32>
  // CHECK: memref.copy
  // CHECK: scf.yield

  %tile = xtile.extract %source[%tile_id][8][1] : memref<64xf32> -> tensor<8xf32>
  %transposed = stablehlo.transpose %tile, dims = [0] : (tensor<8xf32>) -> tensor<8xf32>
  return %transposed : tensor<8xf32>
}

// -----

// Test: identity-layout extract feeding a simple element-wise op (arith.addf)
// with a small tile should yield the subview directly (no alloc in full-tile
// branch).

// CHECK-LABEL: @extract_identity_elementwise
func.func @extract_identity_elementwise(%source: memref<64xf32>, %tile_id: index) -> tensor<8xf32> {
  // 8 x f32 = 32 bytes, well below the 1536-byte register threshold, and
  // arith.addf is element-wise → no forced alloc needed.

  // CHECK: scf.if
  // Full-tile branch: should yield subview directly via to_tensor (no alloc).
  // CHECK: memref.subview
  // CHECK-NOT: memref.alloc
  // CHECK: bufferization.to_tensor
  // CHECK: scf.yield

  %tile = xtile.extract %source[%tile_id][8][1] : memref<64xf32> -> tensor<8xf32>
  %tile2 = xtile.extract %source[%tile_id][8][1] : memref<64xf32> -> tensor<8xf32>
  %result = arith.addf %tile, %tile2 : tensor<8xf32>
  return %result : tensor<8xf32>
}

// -----

// Test: identity-layout extract with a large tile (exceeds register file
// threshold) should force a local buffer even for element-wise users.

// CHECK-LABEL: @extract_identity_large_tile
func.func @extract_identity_large_tile(%source: memref<1024xf32>, %tile_id: index) -> tensor<512xf32> {
  // 512 x f32 = 2048 bytes > 1536-byte threshold → force alloc.

  // CHECK: scf.if
  // Full-tile branch: should alloc + copy due to size.
  // CHECK: memref.subview
  // CHECK: memref.alloc() : memref<512xf32>
  // CHECK: memref.copy
  // CHECK: scf.yield

  %tile = xtile.extract %source[%tile_id][512][1] : memref<1024xf32> -> tensor<512xf32>
  return %tile : tensor<512xf32>
}
