# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pccm

from cumm.common import TensorViewNVRTC, GemmBasic
from cumm.gemm.core.metaarray import MetaArray


class GemmUtilsCPU(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorViewNVRTC, GemmBasic)

    @pccm.static_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def get_logical_tile_count(self):
        code = pccm.code()
        code.arg("m,n,k,tile_m, tile_n, split_k_slices", "int")
        code.ret("tv::array<int, 3>")
        code.raw(f"""
        tv::array<int, 3> grid_dims;
        grid_dims[0] = tv::div_up(m, tile_m);
        grid_dims[1] = tv::div_up(n, tile_n);
        grid_dims[2] = split_k_slices;
        return grid_dims;
        """)
        return code
    
    @pccm.static_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def get_grouped_conv_gemm_logical_tile_count(self):
        code = pccm.code()
        code.arg("m,n,k,groups,tile_m, tile_n, tile_k, split_k_slices", "int")
        code.arg("group_mode", "tv::gemm::ConvGroupMode")
        code.arg("conv_op", "tv::gemm::ConvOpType")
        code.ret("tv::array<int, 3>")
        code.raw(f"""
        tv::array<int, 3> grid_dims;
        if (group_mode == tv::gemm::ConvGroupMode::kNone)
            return get_logical_tile_count(m, n, k, tile_m, tile_n, split_k_slices);
        int c_per_group, k_per_group;
        if (group_mode == tv::gemm::ConvGroupMode::kSingleGroup){{
            if (conv_op == tv::gemm::ConvOpType::kForward){{   // N * gC @ C * gK => N * gK
                assert(k % groups == 0);
                assert(n % groups == 0);
                c_per_group = k / groups;
                k_per_group = n / groups;
                assert(c_per_group % tile_k == 0);
                assert(k_per_group % tile_n == 0);
                assert(split_k_slices == 1);

                grid_dims[0] = tv::div_up(m, tile_m);
                grid_dims[1] = tv::div_up(n, tile_n);
                grid_dims[2] = 1;
                return grid_dims;
            }} else if (conv_op == tv::gemm::ConvOpType::kBackwardInput){{      //N * gK @ gK * C => N * gC
                c_per_group = n;
                assert(k % groups == 0);
                k_per_group = k / groups;
                assert(k_per_group % tile_k == 0);
                assert(c_per_group % tile_n == 0);
                assert(split_k_slices == 1);
                grid_dims[0] = tv::div_up(m, tile_m);
                grid_dims[1] = tv::div_up(n, tile_n);
                grid_dims[2] = groups;
                return grid_dims;
            }} else if (conv_op == tv::gemm::ConvOpType::kBackwardWeight){{     //gK * N @ N * gC => gK * C
                assert(m % groups == 0);
                assert(n % groups == 0);
                k_per_group = m / groups;
                c_per_group = n / groups;
                assert(k_per_group % tile_m == 0);
                assert(c_per_group / tile_n == 0);
                grid_dims[0] = tv::div_up(m, tile_m);
                grid_dims[1] = tv::div_up(c_per_group, tile_n);
                grid_dims[2] = split_k_slices;
                return grid_dims;
            }}
        }} else 
            assert(0);

        grid_dims[0] = tv::div_up(m, tile_m);
        grid_dims[1] = tv::div_up(n, tile_n);
        grid_dims[2] = split_k_slices;
        return grid_dims;
        """)
        return code

    @pccm.static_function(header_only=True, attrs=["TV_HOST_DEVICE_INLINE"])
    def get_gemm_k_size_per_split(self):
        """get gemm per split k
        first we need to get iterations by tile shape k,
        
        """
        code = pccm.code()
        code.arg("k, split_k, tile_k", "int")
        code.raw(f"""
        int total_gemm_k_iterations = tv::div_up(k, tile_k);
        int gemm_k_iterations_per_split =
            tv::div_up(total_gemm_k_iterations, split_k);
        auto gemm_k_size_per_split = gemm_k_iterations_per_split * tile_k; 
        return gemm_k_size_per_split;
        """)
        return code.ret("int")


class GemmUtils(pccm.ParameterizedClass):
    def __init__(self, tile_shape: MetaArray[int]):
        super().__init__()
        self.add_dependency(TensorViewNVRTC, GemmBasic)
        self.tile_shape = tile_shape

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_gemm_k_size_per_split(self):
        """get gemm per split k
        first we need to get iterations by tile shape k,
        
        """
        code = pccm.code()
        code.arg("k, split_k", "int")
        code.raw(f"""
        int total_gemm_k_iterations = tv::div_up(k, {self.tile_shape[2]});
        int gemm_k_iterations_per_split =
            tv::div_up(total_gemm_k_iterations, split_k);
        auto gemm_k_size_per_split = gemm_k_iterations_per_split * {self.tile_shape[2]}; 
        return gemm_k_size_per_split;
        """)
        return code.ret("int")
    
    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_grouped_conv_gemm_k_size_per_split(self):
        code = pccm.code()
        code.arg("k, groups, split_k", "int")
        code.arg("group_mode", "tv::gemm::ConvGroupMode")
        code.arg("conv_op", "tv::gemm::ConvOpType")
        code.ret("int")
        code.raw(f"""
        int total_gemm_k_iterations, gemm_k_iterations_per_split, gemm_k_size_per_split, k_per_group;
        if (group_mode == tv::gemm::ConvGroupMode::kSingleGroup){{
            if (conv_op == tv::gemm::ConvOpType::kForward){{   // N * gC @ C * gK => N * gK
                k_per_group = k / groups;
                assert(split_k == 1);
                total_gemm_k_iterations = tv::div_up(k_per_group, {self.tile_shape[2]});
                gemm_k_size_per_split = total_gemm_k_iterations * {self.tile_shape[2]};
                return gemm_k_size_per_split;

            }} else if (conv_op == tv::gemm::ConvOpType::kBackwardInput){{      //N * gK @ gK * C => N * gC
                k_per_group = k / groups;
                assert(split_k == 1);
                total_gemm_k_iterations = tv::div_up(k_per_group, {self.tile_shape[2]});
                gemm_k_size_per_split = total_gemm_k_iterations * {self.tile_shape[2]};
                return gemm_k_size_per_split;
            }} else if (conv_op == tv::gemm::ConvOpType::kBackwardWeight){{     //gK * N @ N * gC => gK * C
                total_gemm_k_iterations = tv::div_up(k, {self.tile_shape[2]});
                gemm_k_iterations_per_split = tv::div_up(total_gemm_k_iterations, split_k);
                gemm_k_size_per_split = gemm_k_iterations_per_split * {self.tile_shape[2]};
                return gemm_k_size_per_split;
            }}
        }} else 
            assert(0);
        """)
        return code
        

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_gemm_k_bound(self):
        code = pccm.code()
        code.arg("k, gemm_k_size_per_split, tile_offset_k", "int")
        code.raw(f"""
        int k_bound = min(k, (tile_offset_k + 1) * gemm_k_size_per_split);
        return k_bound;
        """)
        return code.ret("int")

    @pccm.cuda.static_function(host=True, device=True, forceinline=True)
    def get_gemm_iterations(self):
        code = pccm.code()
        code.arg("k_bound, gemm_k_size_per_split, tile_offset_k", "int")
        code.raw(f"""
        int gemm_k_iterations =
            tv::div_up(k_bound - tile_offset_k * gemm_k_size_per_split, {self.tile_shape[2]});
        return gemm_k_iterations;
        """)
        return code.ret("int")
