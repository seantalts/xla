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

#include "xla/backends/cpu/runtime/thunk_serdes/slice_to_dynamic_thunk_serdes.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/slice_to_dynamic_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"

namespace xla::cpu {

static absl::Status SliceToDynamicThunkToProto(const Thunk& thunk,
                                               ThunkProto& proto) {
  const auto& slice_to_dynamic =
      absl::down_cast<const SliceToDynamicThunk&>(thunk);
  SliceToDynamicThunkProto* impl = proto.mutable_slice_to_dynamic_thunk();

  ABSL_RETURN_IF_ERROR(SerializeSliceShapeIntoProto(
      slice_to_dynamic.source().slice, slice_to_dynamic.source().shape,
      impl->mutable_source_buffer_shape()));
  for (const BufferAllocation::Slice& dim_size : slice_to_dynamic.dim_sizes()) {
    ABSL_ASSIGN_OR_RETURN(*impl->add_dim_size_buffers(), dim_size.ToProto());
  }
  ABSL_RETURN_IF_ERROR(
      SerializeSliceShapeIntoProto(slice_to_dynamic.destination().slice,
                                   slice_to_dynamic.destination().shape,
                                   impl->mutable_destination_buffer_shape()));
  return absl::OkStatus();
}

static absl::StatusOr<std::unique_ptr<Thunk>> SliceToDynamicThunkFromProto(
    const ThunkProto& proto, const std::vector<BufferAllocation>& allocations,
    const HloModule* hlo_module,
    const std::vector<std::shared_ptr<Resource>>* resources) {
  ABSL_ASSIGN_OR_RETURN(Thunk::Info info, ThunkInfoFromProto(proto.info()));
  const SliceToDynamicThunkProto& impl = proto.slice_to_dynamic_thunk();

  ABSL_ASSIGN_OR_RETURN(
      auto source,
      DeserializeSliceShapeFromProto(impl.source_buffer_shape(), allocations));
  std::vector<BufferAllocation::Slice> dim_sizes;
  for (const auto& dim_size_proto : impl.dim_size_buffers()) {
    ABSL_ASSIGN_OR_RETURN(
        dim_sizes.emplace_back(),
        BufferAllocation::Slice::FromProto(dim_size_proto, allocations));
  }
  ABSL_ASSIGN_OR_RETURN(auto destination,
                        DeserializeSliceShapeFromProto(
                            impl.destination_buffer_shape(), allocations));

  return SliceToDynamicThunk::Create(
      std::move(info), ShapedSlice{source.first, source.second},
      std::move(dim_sizes), ShapedSlice{destination.first, destination.second});
}

void RegisterSliceToDynamicThunkSerDes() {
  CHECK_OK(ThunkSerDesRegistry::Get().Register(Thunk::Kind::kSliceToDynamic,
                                               SliceToDynamicThunkToProto,
                                               SliceToDynamicThunkFromProto));
}

static bool slice_to_dynamic_thunk_serdes_registered = [] {
  RegisterSliceToDynamicThunkSerDes();
  return true;
}();

}  // namespace xla::cpu
