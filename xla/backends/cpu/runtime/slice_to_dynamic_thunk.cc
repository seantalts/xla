/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/slice_to_dynamic_thunk.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<SliceToDynamicThunk>>
SliceToDynamicThunk::Create(Info info, ShapedSlice source,
                            std::vector<BufferAllocation::Slice> dim_sizes,
                            ShapedSlice destination) {
  const Shape& src = source.shape;
  const Shape& dst = destination.shape;
  if (!src.IsArray() || !dst.IsArray()) {
    return InvalidArgument("SliceToDynamic requires array shapes, got %s -> %s",
                           src.ToString(true), dst.ToString(true));
  }
  if (primitive_util::IsSubByteNonPredType(dst.element_type())) {
    return InvalidArgument(
        "SliceToDynamic does not support element type %s",
        primitive_util::LowercasePrimitiveTypeName(dst.element_type()));
  }
  if (!src.is_static()) {
    return InvalidArgument("SliceToDynamic source shape %s must be static",
                           src.ToString(true));
  }
  if (!ShapeUtil::Equal(src, ShapeUtil::MakeStaticShape(dst))) {
    return InvalidArgument(
        "SliceToDynamic source shape %s must match the static bounds of the "
        "destination shape %s",
        src.ToString(true), dst.ToString(true));
  }
  int64_t rank = dst.dimensions().size();
  if (dim_sizes.size() != rank) {
    return InvalidArgument(
        "SliceToDynamic expects %d dimension size operands, got %d", rank,
        dim_sizes.size());
  }
  for (const BufferAllocation::Slice& dim_size : dim_sizes) {
    if (dim_size.size() != sizeof(int32_t)) {
      return InvalidArgument(
          "SliceToDynamic dimension size slice %s must hold one int32",
          dim_size.ToString());
    }
  }
  int64_t required_bytes =
      ShapeUtil::ByteSizeOf(src) + rank * static_cast<int64_t>(sizeof(int32_t));
  if (destination.slice.size() < required_bytes) {
    return InvalidArgument(
        "SliceToDynamic destination slice %s holds %d bytes, needs %d",
        destination.slice.ToString(), destination.slice.size(), required_bytes);
  }
  return absl::WrapUnique(
      new SliceToDynamicThunk(std::move(info), std::move(source),
                              std::move(dim_sizes), std::move(destination)));
}

SliceToDynamicThunk::SliceToDynamicThunk(
    Info info, ShapedSlice source,
    std::vector<BufferAllocation::Slice> dim_sizes, ShapedSlice destination)
    : Thunk(Kind::kSliceToDynamic, std::move(info)),
      source_(std::move(source)),
      dim_sizes_(std::move(dim_sizes)),
      destination_(std::move(destination)),
      element_size_(
          ShapeUtil::ByteSizeOfPrimitiveType(source_.shape.element_type())),
      raw_data_size_(ShapeUtil::ByteSizeOf(source_.shape)) {}

Thunk::BufferUses SliceToDynamicThunk::buffer_uses() const {
  BufferUses uses;
  uses.push_back(BufferUse::Read(source_.slice, source_.shape));
  for (const BufferAllocation::Slice& dim_size : dim_sizes_) {
    uses.push_back(BufferUse::Read(dim_size, ShapeUtil::MakeScalarShape(S32)));
  }
  uses.push_back(BufferUse::Write(destination_.slice, destination_.shape));
  return uses;
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> SliceToDynamicThunk::Execute(
    const ExecuteParams& params) {
  const BufferAllocations* allocations = params.buffer_allocations;
  const Shape& shape = source_.shape;
  const int64_t rank = shape.dimensions().size();

  se::DeviceAddressBase source_data;
  se::DeviceAddressBase destination_data;
  absl::InlinedVector<se::DeviceAddressBase, 4> dim_data(rank);
  if constexpr (ShouldCheckBufferSlices()) {
    ABSL_ASSIGN_OR_RETURN(source_data,
                          allocations->GetDeviceAddress(source_.slice));
    ABSL_ASSIGN_OR_RETURN(destination_data,
                          allocations->GetDeviceAddress(destination_.slice));
    for (int64_t i = 0; i < rank; ++i) {
      ABSL_ASSIGN_OR_RETURN(dim_data[i],
                            allocations->GetDeviceAddress(dim_sizes_[i]));
    }
  } else {
    source_data = allocations->GetDeviceAddressUnchecked(source_.slice);
    destination_data =
        allocations->GetDeviceAddressUnchecked(destination_.slice);
    for (int64_t i = 0; i < rank; ++i) {
      dim_data[i] = allocations->GetDeviceAddressUnchecked(dim_sizes_[i]);
    }
  }

  const std::byte* source = static_cast<const std::byte*>(source_data.opaque());
  std::byte* destination = static_cast<std::byte*>(destination_data.opaque());

  absl::InlinedVector<int64_t, 4> extents(rank);
  for (int64_t i = 0; i < rank; ++i) {
    int32_t extent;
    std::memcpy(&extent, dim_data[i].opaque(), sizeof(int32_t));
    if (extent < 0 || extent > shape.dimensions(i)) {
      return InvalidArgument(
          "SliceToDynamic extent %d of dimension %d is outside [0, %d] for %s",
          extent, i, shape.dimensions(i), shape.ToString(true));
    }
    std::memcpy(destination + raw_data_size_ + i * sizeof(int32_t), &extent,
                sizeof(int32_t));
    extents[i] = extent;
  }

  if (rank == 0) {
    std::memcpy(destination, source, element_size_);
    return OkExecuteEvent();
  }

  int64_t num_elements = 1;
  for (int64_t extent : extents) {
    num_elements *= extent;
  }
  if (num_elements == 0) {
    return OkExecuteEvent();
  }

  absl::Span<const int64_t> minor_to_major = shape.layout().minor_to_major();
  const int64_t run_bytes = extents[minor_to_major[0]] * element_size_;

  // Strides in elements, indexed by logical dimension: the source keeps the
  // static bounds, the packed destination uses the dynamic extents.
  absl::InlinedVector<int64_t, 4> source_strides(rank);
  absl::InlinedVector<int64_t, 4> destination_strides(rank);
  int64_t source_stride = 1;
  int64_t destination_stride = 1;
  for (int64_t dim : minor_to_major) {
    source_strides[dim] = source_stride;
    destination_strides[dim] = destination_stride;
    source_stride *= shape.dimensions(dim);
    destination_stride *= extents[dim];
  }

  absl::InlinedVector<int64_t, 4> index(rank, 0);
  int64_t source_offset = 0;
  int64_t destination_offset = 0;
  while (true) {
    std::memcpy(destination + destination_offset * element_size_,
                source + source_offset * element_size_, run_bytes);

    int64_t i = 1;
    for (; i < rank; ++i) {
      int64_t dim = minor_to_major[i];
      source_offset += source_strides[dim];
      destination_offset += destination_strides[dim];
      if (++index[dim] < extents[dim]) {
        break;
      }
      source_offset -= source_strides[dim] * extents[dim];
      destination_offset -= destination_strides[dim] * extents[dim];
      index[dim] = 0;
    }
    if (i == rank) {
      break;
    }
  }

  return OkExecuteEvent();
}

}  // namespace xla::cpu
