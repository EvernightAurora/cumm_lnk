// Copyright 2022 Yan Yan
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

#pragma once
#include "constants.h"
#include <tensorview/core/all.h>
namespace tv {
namespace gemm {
TV_HOST_DEVICE_INLINE tv::array<int, 3>
get_logical_tile_count(int m, int n, int k, int tile_m, int tile_n,
                       int split_k_slices) {
  tv::array<int, 3> grid_dims;
  grid_dims[0] = tv::div_up(m, tile_m);
  grid_dims[1] = tv::div_up(n, tile_n);
  grid_dims[2] = split_k_slices;
  return grid_dims;
}

TV_HOST_DEVICE_INLINE int get_gemm_k_size_per_split(int k, int split_k,
                                                    int tile_k) {
  int total_gemm_k_iterations = tv::div_up(k, tile_k);
  int gemm_k_iterations_per_split =
      tv::div_up(total_gemm_k_iterations, split_k);
  auto gemm_k_size_per_split = gemm_k_iterations_per_split * tile_k;
  return gemm_k_size_per_split;
}

TV_HOST_DEVICE_INLINE tv::array<int, 3>
get_spconv_logical_tile_count(int m, int n, int k, int tile_m, int tile_n,
                              int split_k_slices, int kv, ConvOpType op_type) {
  tv::array<int, 3> grid_dims;
  if (op_type == ConvOpType::kBackwardWeight) {
    {
      // n = C * kv
      int C = n / kv;
      // for wgrad, we need to ensure a block must be covered by one mask
      // so refined_n = tv::div_up(C, tile_n) * tile_n * kv
      // for example, C = 130, tile_n = 64, so one kernel loc need three
      // block 64 * 3 = 192, then refined_n = 192 * kv
      // n = tv::div_up(C, tile_n) * tile_n * kv;
      grid_dims[1] = tv::div_up(C, tile_n) * kv;
    }
  } else {
    { grid_dims[1] = tv::div_up(n, tile_n); }
  }
  grid_dims[0] = tv::div_up(m, tile_m);
  grid_dims[2] = split_k_slices;
  return grid_dims;
}

TV_HOST_DEVICE_INLINE tv::array<int, 3>
get_single_grouped_spconv_logical_tile_count(int m, int n, int k, int tile_m, int tile_n,
                              int split_k_slices, int kv, ConvOpType op_type,
                              int groups, int C_per_group, int K_per_group) {
  tv::array<int, 3> grid_dims;
  switch (op_type){
      case ConvOpType::kForward:
          grid_dims[0] = tv::div_up(m, tile_m);
          grid_dims[1] = K_per_group / tile_n * groups;
          grid_dims[2] = split_k_slices;                  // should be 1
          break;

      case ConvOpType::kBackwardInput:
          grid_dims[0] = tv::div_up(m, tile_m);
          grid_dims[1] = C_per_group / tile_n;
          grid_dims[2] = groups;
          break;
      
      case ConvOpType::kBackwardWeight:
          grid_dims[0] = K_per_group / tile_m * groups;
          grid_dims[1] = (C_per_group / tile_n) * kv;
          grid_dims[2] = split_k_slices;
          break;
      default:
          assert(0);
  }
  return grid_dims;
}

TV_HOST_DEVICE_INLINE tv::array<int, 3>
get_depthwise_spconv_logical_tile_count(int m, int n, int k, int tile_m, int tile_n,
                              int split_k_slices, int kv, ConvOpType op_type) {
  tv::array<int, 3> grid_dims;
  if (op_type == ConvOpType::kForward)
    return get_spconv_logical_tile_count(m, n, k, tile_m, tile_n, split_k_slices, kv, op_type);
  else
    assert(0);
  return tv::array<int, 3>();
}

TV_HOST_DEVICE_INLINE tv::array<int, 3>
get_unaligned_single_grouped_spconv_logical_tile_count(int m, int n, int k, int tile_m, int tile_n,
                              int split_k_slices, int kv, ConvOpType op_type,
                              int groups, int C_per_group, int K_per_group,
                              int tiles_per_C, int tiles_per_K) {
  tv::array<int, 3> grid_dims;
  switch (op_type){
      case ConvOpType::kForward:
          grid_dims[0] = tv::div_up(m, tile_m);
          grid_dims[1] = tiles_per_K * groups;
          grid_dims[2] = split_k_slices;                  // should be 1
          break;

      case ConvOpType::kBackwardInput:
          grid_dims[0] = tv::div_up(m, tile_m);
          grid_dims[1] = tiles_per_C;
          grid_dims[2] = groups;
          break;
      
      case ConvOpType::kBackwardWeight:
          grid_dims[0] = tiles_per_K * groups;
          grid_dims[1] = tiles_per_C * kv;
          grid_dims[2] = split_k_slices;
          break;
      default:
          assert(0);
  }
  return grid_dims;
}


TV_HOST_DEVICE_INLINE tv::array<int, 3>
implicit_gemm_mnk(tv::gemm::ConvOpType op_type, int N, int C, int K,
                  int kernel_volume, int in_prod, int out_prod,
                  bool mask_sparse) {

  if (mask_sparse) {
    switch (op_type) {
    case tv::gemm::ConvOpType::kForward:
      return {N, K, C * kernel_volume};
    case tv::gemm::ConvOpType::kBackwardInput:
      return {N, C, K * kernel_volume};
    case tv::gemm::ConvOpType::kBackwardWeight:
      return {K, C * kernel_volume, N};
    default:
      return {};
    }
    return {};
  } else {
    switch (op_type) {
    case tv::gemm::ConvOpType::kForward:
      return {N * out_prod, K, C * kernel_volume};
    case tv::gemm::ConvOpType::kBackwardInput:
      return {N * in_prod, C, K * kernel_volume};
    case tv::gemm::ConvOpType::kBackwardWeight:
      return {K, C * kernel_volume, N * out_prod};
    default:
      return {};
    }
    return {};
  }
}
TV_HOST_DEVICE_INLINE tv::array<int, 3>
conv_iwo_012_to_abc(tv::gemm::ConvOpType op_type) {

  switch (op_type) {
  case tv::gemm::ConvOpType::kForward:
    return {0, 1, 2};
  case tv::gemm::ConvOpType::kBackwardInput:
    return {2, 1, 0};
  case tv::gemm::ConvOpType::kBackwardWeight:
    return {1, 2, 0};
  default:
    return {};
  }
  return {};
}
TV_HOST_DEVICE_INLINE tv::array<int, 3>
gemm_abc_012_to_iwo(tv::gemm::ConvOpType op_type) {

  switch (op_type) {
  case tv::gemm::ConvOpType::kForward:
    return {0, 1, 2};
  case tv::gemm::ConvOpType::kBackwardInput:
    return {2, 1, 0};
  case tv::gemm::ConvOpType::kBackwardWeight:
    return {2, 0, 1};
  default:
    return {};
  }
  return {};
}

} // namespace gemm
} // namespace tv