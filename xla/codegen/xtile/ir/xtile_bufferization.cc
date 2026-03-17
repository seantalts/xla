/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

#include "absl/log/check.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/implicit_arith_op_builder.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::xtile {

// Vector register file capacity threshold for deciding whether a tile should
// be forced into a local stack buffer. AVX-512 has 32 x 64-byte registers =
// 2048 bytes; we use 75% to leave headroom for intermediate values and
// loop-carried state.
static constexpr int64_t kVectorRegisterFileThresholdBytes = 1536;

// Returns true if `op` is a data-layout-altering operation such as transpose
// or broadcast. These operations have non-sequential memory access patterns
// that perform poorly when operating directly on strided buffer subviews
// (cache-line bouncing, register spill).
static bool IsLayoutAlteringOp(mlir::Operation* op) {
  llvm::StringRef name = op->getName().getStringRef();

  // Explicit transpose or broadcast operations from any dialect.
  if (name == "linalg.transpose" || name == "vector.transpose" ||
      name == "linalg.broadcast" || name == "vector.broadcast" ||
      name == "stablehlo.transpose" ||
      name == "stablehlo.broadcast_in_dim") {
    return true;
  }

  // For linalg.generic, check if input indexing maps differ from the output
  // map, which indicates a data-layout transformation (e.g., the input is
  // accessed in transposed order relative to the output).
  if (name == "linalg.generic") {
    auto indexing_maps_attr =
        op->getAttrOfType<mlir::ArrayAttr>("indexing_maps");
    if (!indexing_maps_attr || indexing_maps_attr.size() < 2) {
      return false;
    }

    // The last map is the output; compare every input map against it.
    auto output_map =
        mlir::cast<mlir::AffineMapAttr>(indexing_maps_attr.getValue().back())
            .getValue();
    for (auto map_attr : indexing_maps_attr.getValue().drop_back()) {
      auto input_map = mlir::cast<mlir::AffineMapAttr>(map_attr).getValue();
      // A map with fewer results than the output indicates broadcast;
      // a map with the same results but different ordering indicates
      // transpose / permutation.
      if (input_map != output_map) {
        return true;
      }
    }
  }

  return false;
}

// Returns true if any direct user of `value` is a layout-altering op.
static bool HasLayoutAlteringUser(mlir::Value value) {
  for (mlir::Operation* user : value.getUsers()) {
    if (IsLayoutAlteringOp(user)) {
      return true;
    }
  }
  return false;
}

// Conservative estimate of a tile's memory footprint in bytes. Returns
// int64_t max for dynamic shapes so the caller treats them as "large".
static int64_t EstimateTileFootprintBytes(mlir::RankedTensorType type) {
  if (!type.hasStaticShape()) {
    return std::numeric_limits<int64_t>::max();
  }
  int64_t num_elements = 1;
  for (int64_t dim : type.getShape()) {
    num_elements *= dim;
  }
  int64_t element_bits = type.getElementTypeBitWidth();
  return num_elements * ((element_bits + 7) / 8);
}

// Decides whether `op` should force a contiguous local buffer allocation
// instead of yielding a (possibly strided) subview directly. This prevents
// performance regressions for:
//   1. Layout-altering ops (transpose, broadcast) that have non-sequential
//      access patterns on strided subviews → poor cache utilisation.
//   2. Large tiles that exceed the vector register file → register spill and
//      SROA (Scalar Replacement of Aggregates) failure.
static bool ShouldForceLocalBuffer(ExtractTileOp op) {
  if (HasLayoutAlteringUser(op.getResult())) {
    return true;
  }
  if (EstimateTileFootprintBytes(op.getResult().getType()) >
      kVectorRegisterFileThresholdBytes) {
    return true;
  }
  return false;
}

static llvm::SmallVector<mlir::OpFoldResult> GetStaticFoldResult(
    mlir::OpBuilder& builder, llvm::ArrayRef<int64_t> input) {
  return llvm::map_to_vector(input, [&builder](int64_t value) {
    return mlir::OpFoldResult(builder.getIndexAttr(value));
  });
}

static llvm::SmallVector<mlir::OpFoldResult> GetDynamicFoldResult(
    mlir::ValueRange input) {
  return llvm::SmallVector<mlir::OpFoldResult>(input);
}

// Get the size of the memref subview with the output size clamped to inbound
// elements, if full_size is true then unit values are inserted for reduced
// dimensions.
// The derivation of these bounds is as follows:
//   index + tile_size * stride <= size - 1
//   tile_size * stride <= size - 1 - index
//   tile_size <= size - 1 - index / stride
//   tile_size < ((size - 1 - index) / stride) + 1
static llvm::SmallVector<mlir::OpFoldResult> GetClampedTileSize(
    mlir::ImplicitLocOpBuilder& builder, TiledBufferInterface op,
    bool full_size) {
  llvm::SmallVector<mlir::OpFoldResult> tile_size;
  llvm::SmallDenseSet<unsigned> reduced_dims = op.getReducedDimensions();
  int64_t idx = 0;
  for (auto [buffer_size, offset, stride, full_tile_size] :
       llvm::zip(op.getBuffer().getType().getShape(), op.getOffsets(),
                 op.getStrides(), op.getFullTileShape())) {
    if (reduced_dims.contains(idx++)) {
      if (full_size) {
        tile_size.emplace_back(builder.getIndexAttr(1));
      }
      continue;
    }
    emitters::ImplicitArithOpBuilder arith_builder(
        mlir::arith::ConstantIndexOp::create(builder, (buffer_size - 1)),
        &builder);
    auto numerator = arith_builder - offset;
    // The stride can be 0 for single element tiles.
    // TODO(willfroom): Fix tile analysis so this never happens.
    auto clamped_stride = std::max<int64_t>(stride, 1);
    auto bound = numerator / clamped_stride + 1;
    tile_size.emplace_back(bound.min(full_tile_size));
  }

  return tile_size;
}

// Get the subview of the op buffer with its size clamped such that all elements
// are in bounds.
static mlir::TypedValue<mlir::MemRefType> GetClampedSubView(
    mlir::ImplicitLocOpBuilder& builder, TiledBufferInterface op) {
  auto tile_size = GetClampedTileSize(builder, op, true);

  auto offsets = GetDynamicFoldResult(op.getOffsets());
  auto strides = GetStaticFoldResult(builder, op.getStrides());

  mlir::RankedTensorType tile_type = op.getTile().getType();
  llvm::SmallVector<int64_t> output_shape(tile_type.getRank(),
                                          mlir::ShapedType::kDynamic);
  mlir::MemRefType subview_type =
      mlir::memref::SubViewOp::inferRankReducedResultType(
          output_shape, op.getBuffer().getType(), offsets, tile_size, strides);

  return mlir::memref::SubViewOp::create(builder, subview_type, op.getBuffer(),
                                         offsets, tile_size, strides);
}

// Gets the subview of the op buffer with the precondition that the tile fits
// within the buffer.
static mlir::TypedValue<mlir::MemRefType> GetFullTileSubView(
    mlir::ImplicitLocOpBuilder& builder, TiledBufferInterface op) {
  auto offsets = GetDynamicFoldResult(op.getOffsets());
  auto tile_size = GetStaticFoldResult(builder, op.getFullTileShape());
  auto strides = GetStaticFoldResult(builder, op.getStrides());

  mlir::RankedTensorType tile_type = op.getTile().getType();
  mlir::MemRefType subview_type =
      mlir::memref::SubViewOp::inferRankReducedResultType(
          tile_type.getShape(), op.getBuffer().getType(), offsets, tile_size,
          strides);

  return mlir::memref::SubViewOp::create(builder, subview_type, op.getBuffer(),
                                         offsets, tile_size, strides);
}

// Get the subview of the local buffer - i.e it has 0 offsets & unit strides.
static mlir::TypedValue<mlir::MemRefType> GetLocalBufferSubview(
    mlir::ImplicitLocOpBuilder& builder,
    mlir::TypedValue<mlir::MemRefType> buffer,
    llvm::ArrayRef<mlir::OpFoldResult> tile_size,
    llvm::ArrayRef<int64_t> full_tile_shape) {
  mlir::SmallVector<mlir::OpFoldResult> buffer_offsets(
      buffer.getType().getRank(), builder.getIndexAttr(0));
  mlir::SmallVector<mlir::OpFoldResult> buffer_strides(
      buffer.getType().getRank(), builder.getIndexAttr(1));

  mlir::MemRefType buffer_subview_type =
      mlir::memref::SubViewOp::inferRankReducedResultType(
          full_tile_shape, buffer.getType(), buffer_offsets, tile_size,
          buffer_strides);
  return mlir::memref::SubViewOp::create(builder, buffer_subview_type, buffer,
                                         buffer_offsets, tile_size,
                                         buffer_strides);
}

// Extract the slice of the tensor that is clamped to be within bounds of the
// target buffer.
static mlir::TypedValue<mlir::RankedTensorType> GetTensorSlice(
    mlir::ImplicitLocOpBuilder& builder, InsertTileOp op) {
  auto tile_size = GetClampedTileSize(builder, op, false);

  mlir::SmallVector<mlir::OpFoldResult> offsets(tile_size.size(),
                                                builder.getIndexAttr(0));
  mlir::SmallVector<mlir::OpFoldResult> strides(tile_size.size(),
                                                builder.getIndexAttr(1));

  return mlir::tensor::ExtractSliceOp::create(builder, op.getSource(), offsets,
                                              tile_size, strides);
}

static mlir::Value TileIsFullSize(mlir::ImplicitLocOpBuilder& builder,
                                  TiledBufferInterface op) {
  llvm::SmallVector<mlir::OpFoldResult> clamped_tile_size =
      GetClampedTileSize(builder, op, false);
  mlir::Value is_full_size =
      mlir::arith::ConstantIntOp::create(builder, builder.getI1Type(), true);
  for (auto [dim_idx, tile_dim_size] : llvm::enumerate(clamped_tile_size)) {
    if (auto value = tile_dim_size.dyn_cast<mlir::Value>()) {
      mlir::Value is_full_size_dim = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::eq, value,
          mlir::arith::ConstantIndexOp::create(
              builder, op.getTile().getType().getDimSize(dim_idx)));
      is_full_size =
          mlir::arith::AndIOp::create(builder, is_full_size, is_full_size_dim);
    }
  }
  return is_full_size;
}

// Get a buffer copied from the original buffer that is padded to the full tile
// size.
static mlir::TypedValue<mlir::MemRefType> GetPaddedTileBuffer(
    mlir::ImplicitLocOpBuilder& builder, ExtractTileOp op) {
  auto buffer_tile_subview = GetClampedSubView(builder, op);
  mlir::RankedTensorType tile_type = op.getResult().getType();
  auto buffer = mlir::memref::AllocOp::create(
      builder, GetStaticFoldResult(builder, tile_type.getShape()),
      tile_type.getElementType());

  auto local_tile_size = GetClampedTileSize(builder, op, false);
  auto local_buffer_subview =
      GetLocalBufferSubview(builder, buffer, local_tile_size,
                            buffer_tile_subview.getType().getShape());

  mlir::memref::CopyOp::create(builder, buffer_tile_subview,
                               local_buffer_subview);

  return buffer;
}

bool ExtractTileOp::bufferizesToMemoryRead(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return true;
}

bool ExtractTileOp::bufferizesToMemoryWrite(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return true;
}

bool ExtractTileOp::bufferizesToAllocation(mlir::Value value) {
  // As we don't know if we will emit an allocation at compile time we must be
  // conservative.
  return true;
}

mlir::bufferization::AliasingValueList ExtractTileOp::getAliasingValues(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return {};
}

mlir::bufferization::AliasingOpOperandList ExtractTileOp::getAliasingOpOperands(
    mlir::Value value, const mlir::bufferization::AnalysisState& state) {
  DCHECK_EQ(value, getResult());
  mlir::bufferization::AliasingOpOperand result(
      &getSourceMutable(), mlir::bufferization::BufferRelation::Equivalent,
      false);
  return {result};
}

bool ExtractTileOp::isWritable(
    mlir::Value value, const mlir::bufferization::AnalysisState& state) {
  return false;
}

llvm::LogicalResult ExtractTileOp::bufferize(
    mlir::RewriterBase& rewriter,
    const mlir::bufferization::BufferizationOptions& options,
    mlir::bufferization::BufferizationState& state) {
  mlir::ImplicitLocOpBuilder builder(getLoc(), rewriter);

  mlir::Value is_full_size = TileIsFullSize(builder, *this);
  auto if_op = mlir::scf::IfOp::create(
      builder, is_full_size,
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder then_builder(loc, builder);
        auto buffer = GetFullTileSubView(then_builder, *this);
        bool force_local = ShouldForceLocalBuffer(*this);
        if (buffer.getType().getLayout().isIdentity() && !force_local) {
          // Fast path: the subview is contiguous and the downstream compute
          // is simple element-wise with a small tile. Allow the one-shot
          // bufferizer to fold through this tensor without an extra copy.
          auto to_tensor_op = mlir::bufferization::ToTensorOp::create(
              then_builder, getType(), buffer);
          mlir::scf::YieldOp::create(then_builder, {to_tensor_op});
        } else {
          // Slow path: either the layout is non-identity (strided subview)
          // or the heuristic determined we need a local buffer because:
          //   - A downstream op alters data layout (transpose, broadcast)
          //     and would perform poorly on a strided/aliased subview, or
          //   - The tile exceeds the vector register file threshold and
          //     keeping it in a subview would cause register spill / SROA
          //     failure.
          // Allocate a contiguous local buffer and copy into it.
          mlir::MemRefType default_buffer_type =
              mlir::MemRefType::Builder(buffer.getType()).setLayout(nullptr);
          auto default_buffer =
              mlir::memref::AllocOp::create(then_builder, default_buffer_type);
          mlir::memref::CopyOp::create(then_builder, buffer, default_buffer);
          auto to_tensor_op = mlir::bufferization::ToTensorOp::create(
              then_builder, getType(), default_buffer);
          to_tensor_op.setWritable(true);
          to_tensor_op.setRestrict(true);
          mlir::scf::YieldOp::create(then_builder, {to_tensor_op});
        }
      },
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder else_builder(loc, builder);
        auto buffer = GetPaddedTileBuffer(else_builder, *this);
        auto to_tensor_op = mlir::bufferization::ToTensorOp::create(
            else_builder, getType(), buffer);
        to_tensor_op.setWritable(true);
        to_tensor_op.setRestrict(true);
        mlir::scf::YieldOp::create(else_builder, {to_tensor_op});
      });

  rewriter.replaceOp(getOperation(), if_op.getResults());

  return mlir::success();
}

bool InsertTileOp::bufferizesToMemoryRead(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return true;
}

bool InsertTileOp::bufferizesToMemoryWrite(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  DCHECK_EQ(operand.getOperandNumber(), 0)
      << "This should only be called on the tensor operand.";
  return false;
}

bool InsertTileOp::bufferizesToAllocation(mlir::Value value) {
  // As we don't know if we will emit an allocation at compile time we must be
  // conservative.
  return true;
}

mlir::bufferization::AliasingValueList InsertTileOp::getAliasingValues(
    mlir::OpOperand& operand, const mlir::bufferization::AnalysisState& state) {
  return {};
}

mlir::bufferization::AliasingOpOperandList InsertTileOp::getAliasingOpOperands(
    mlir::Value value, const mlir::bufferization::AnalysisState& state) {
  return {};
}

bool InsertTileOp::isWritable(mlir::Value value,
                              const mlir::bufferization::AnalysisState& state) {
  if (value == getDestination()) {
    return true;
  }

  return false;
}

llvm::LogicalResult InsertTileOp::bufferize(
    mlir::RewriterBase& rewriter,
    const mlir::bufferization::BufferizationOptions& options,
    mlir::bufferization::BufferizationState& state) {
  mlir::ImplicitLocOpBuilder builder(getLoc(), rewriter);

  mlir::Value is_full_size = TileIsFullSize(builder, *this);
  mlir::scf::IfOp::create(
      builder, is_full_size,
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder then_builder(loc, builder);
        auto target_buffer_subview = GetFullTileSubView(then_builder, *this);
        auto materialize_op =
            mlir::bufferization::MaterializeInDestinationOp::create(
                then_builder, getSource(), target_buffer_subview);
        materialize_op.setWritable(true);
        mlir::scf::YieldOp::create(then_builder);
      },
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        mlir::ImplicitLocOpBuilder else_builder(loc, builder);
        auto tile_slice = GetTensorSlice(else_builder, *this);
        auto target_buffer_subview = GetClampedSubView(else_builder, *this);
        auto materialize_op =
            mlir::bufferization::MaterializeInDestinationOp::create(
                else_builder, tile_slice, target_buffer_subview);
        materialize_op.setWritable(true);
        mlir::scf::YieldOp::create(else_builder);
      });

  rewriter.eraseOp(getOperation());
  return mlir::success();
}

}  // namespace xla::xtile
