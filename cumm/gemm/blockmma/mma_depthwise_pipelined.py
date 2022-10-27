# Copyright 2022 Yan Yan, Fan Xie
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

import contextlib
from typing import Dict, List, Optional, Tuple, Type, Union, overload

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, GemmBasicKernel, TensorViewKernel
from cumm.conv.bases import ConvOpType
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import (constants, layout, mask_iters, out_iters, thread_map,
                       volta_iters, volta_out_iters)
from cumm.gemm.algospec import bases
from cumm.gemm.arch.memory import GlobalLoad
from cumm.gemm.bases import (GemmInputIterator, GemmOutputIterator,
                             GemmOutputOp, GemmOutSmemLoader,
                             GemmOutWarpIterator, GemmSmemIterator,
                             GemmWarpIterator, GemmComponentBase)
from cumm.gemm.core import MetaArray, array_type, metaseq, seq
from .mma import Mma, BlockMmaStorage, MaskIGemmIterator


class DepthwiseWarpIterArrangement(GemmComponentBase):      # for lane_mma_shape[1] <= tileK <= lane_mma_shape[1] * warp_shape[1]
    def __init__(self, mma_spec: bases.Mma) -> None:
        super().__init__()
        self.tK = mma_spec.input_spec.tile_shape[2]
        self.tN = mma_spec.input_spec.tile_shape[1]
        self.gemm_k_iters = self.tN // self.tK
        self.thread_mma_n = mma_spec.thread_mma_shape[1]
        self.lane_mma_n = mma_spec.lane_mma_shape[1]
        self.warp_shape_n = mma_spec.warp_shape[1]
        assert self.satisfied(mma_spec)
        self.add_member("lane_n_start", "int")
        self.add_dependency(layout.RowMajor,
                            layout.ColumnMajor)
        self.mma_spec = mma_spec
    
    @staticmethod
    def satisfied(mma_spec: bases.Mma):
        if not mma_spec.input_spec.warp_count_shape[1] == 1:
            return False
        return mma_spec.lane_mma_shape[1] <= mma_spec.input_spec.tile_shape[2] <= mma_spec.lane_mma_shape[1] * mma_spec.warp_shape[1]
    
    @property
    def warp_mma_iters(self):
        return self.lane_mma_n
    
    def get_cycle(self):
        return self.lane_mma_n * self.gemm_k_iters
    
    @pccm.cuda.constructor(device=True, forceinline=True) 
    def ctor(self):
        code = pccm.code()
        code.arg("lane_idx", "int")
        code.raw(f"""
            auto layoutW = RowMajor::from_shape({{{self.mma_spec.warp_shape[0]}, {self.mma_spec.warp_shape[1]}}});
            lane_n_start = (layoutW.inverse_1(lane_idx) * {self.lane_mma_n}) % {self.tK};
        """)
        return code
    
    @pccm.cuda.member_function(device=True, forceinline=True, name="operator()")
    def call(self):
        code = pccm.code()
        code.targ("T")
        code.arg("warp_iter", "T&")
        code.arg("gemm_k_iterations, warp_mma_k", "int", "-1")
        code.raw(f"""
            if (gemm_k_iterations < 0){{
                warp_iter.set_kgroup_index(0);
                warp_iter.tile_increment(lane_n_start);
                warp_iter.set_kgroup_index(lane_n_start);
                return;
            }}
            if (warp_mma_k == {self.warp_mma_iters - 1}){{
                warp_iter.set_kgroup_index(({self.tK} * ((gemm_k_iterations + 1) % {self.tN // self.tK})));
                warp_iter.tile_increment({self.tK} - {self.warp_mma_iters - 1});
                warp_iter.set_kgroup_index(lane_n_start + ({self.tK} * ((gemm_k_iterations + 1) % {self.tN // self.tK})));
                return;
            }}
            ++warp_iter;
            warp_iter.set_kgroup_index(lane_n_start + ({self.tK} * gemm_k_iterations) + warp_mma_k + 1);
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def valid_b_idx(self):
        code = pccm.code()
        code.arg("b_valid_idx", "int&")
        code.arg("gemm_k_iterations, warp_mma_k", "int", "-1")
        code.raw(f"""
            int group_idx;
            if (gemm_k_iterations < 0){{
                group_idx = lane_n_start;
            }}
            else if (warp_mma_k == {self.warp_mma_iters - 1}){{
                group_idx = lane_n_start + ({self.tK} * ((gemm_k_iterations + 1) % {self.tN // self.tK}));
            }} else{{
                group_idx = lane_n_start + ({self.tK} * gemm_k_iterations) + warp_mma_k + 1;
            }}
            b_valid_idx = (group_idx % {self.lane_mma_n}) + (group_idx / {self.lane_mma_n * self.warp_shape_n}) * {self.lane_mma_n};
        """)
        return code

    @pccm.cuda.member_function(device=True, forceinline=True)
    def increment_and_calc(self):
        code = pccm.code()
        code.targ("WA")
        code.targ("WB")
        code.arg("warp_iter_a", "WA&")
        code.arg("warp_iter_b", "WB&")
        code.arg("b_valid_idx", "int&")
        code.arg("gemm_k_iterations, warp_mma_k", "int", "-1")
        code.raw(f"""
            int kgroup_idx;
            if (gemm_k_iterations < 0){{
                warp_iter_b.set_kgroup_index(0);
                warp_iter_b.tile_increment(lane_n_start);
                warp_iter_a.tile_increment(lane_n_start);
                kgroup_idx = lane_n_start;
            }}else if (warp_mma_k == {self.warp_mma_iters - 1}){{
                warp_iter_b.set_kgroup_index(({self.tK} * ((gemm_k_iterations + 1) % {self.tN // self.tK})));
                warp_iter_b.tile_increment({self.tK} - {self.warp_mma_iters - 1});
                warp_iter_a.tile_increment({self.tK} - {self.warp_mma_iters - 1});
                kgroup_idx = lane_n_start + ({self.tK} * ((gemm_k_iterations + 1) % {self.tN // self.tK}));
            }} else{{
                ++warp_iter_a;
                ++warp_iter_b;
                kgroup_idx = lane_n_start + ({self.tK} * gemm_k_iterations) + warp_mma_k + 1;
            }}
            warp_iter_b.set_kgroup_index(kgroup_idx);
            b_valid_idx = (kgroup_idx % {self.lane_mma_n}) + (kgroup_idx / {self.lane_mma_n * self.warp_shape_n}) * {self.lane_mma_n};
        """)
        return code



class MmaDepthwiseConvPipelined(GemmComponentBase):  # TODO:  to inhret from Mma
    def __init__(self,
                 dtype_acc: dtypes.DType,
                 partk: int,
                 num_stage: int,
                 spec: bases.Mma,
                 smem_storage: BlockMmaStorage,
                 first_input_clear: bool = True,
                 clear_mask: bool = True,
                 mask_sparse: bool = False,
                 increment_k_first: bool =False,
                 op_type: ConvOpType =  ConvOpType.kForward):
        self.op_type = op_type
        is_sparse_wgrad = op_type == ConvOpType.kBackwardWeight
        # super().__init__(dtype_acc, partk, num_stage, spec, smem_storage, first_input_clear,
        #                     clear_mask, mask_sparse, increment_k_first, is_sparse_wgrad)
        super().__init__()

        self.dtype_acc = dtype_acc
        miter = MaskIGemmIterator(increment_k_first)
        self.add_param_class("mma_ns_miter", miter, "MaskIGemmIterator")

        self.add_param_class("mma_ns_wa", spec.warp_iter_a, "WarpIterA")
        self.add_param_class("mma_ns_wb", spec.warp_iter_b, "WarpIterB")
        self.add_param_class("mma_ns_sa", spec.smem_iter_a, "SmemIterA")
        self.add_param_class("mma_ns_sb", spec.smem_iter_b, "SmemIterB")
        self.smem_storage = smem_storage
        self.spec = spec
        self.num_stage = num_stage
        self.mask_sparse = mask_sparse
        self.increment_k_first = increment_k_first
        self.partk = partk
        self.first_input_clear = first_input_clear
        self.clear_mask = clear_mask
        self.input_spec = spec.input_spec
        self.is_sparse_wgrad = is_sparse_wgrad
        if is_sparse_wgrad:
            self.add_param_class("gl_wgrad", GlobalLoad(4), "GlobalLoad")
        self.add_param_class("mma_ns_gm", smem_storage, "GemmStorage")
        self.accumulator_fragment = array_type(dtype_acc,
                                               spec.accumulator_size)
        self.add_param_class("mma_ns_ia", self.input_spec.input_iter_a,
                             "InputIteratorA")
        self.add_param_class("mma_ns_ib", self.input_spec.input_iter_b,
                             "InputIteratorB")
        self.add_param_class("mma_ns_wmma", spec.warp_mma, "WarpMma")
        self.wmma = spec.warp_mma
        self.add_member("warp_iter_A", "WarpIterA")
        self.add_member("warp_iter_B", "WarpIterB")
        self.add_member("smem_iter_A", "SmemIterA")
        self.add_member("smem_iter_B", "SmemIterB")

        # cudasim
        self.warp_iter_A: Optional[GemmWarpIterator] = None
        self.warp_iter_B: Optional[GemmWarpIterator] = None
        self.smem_iter_A: Optional[GemmSmemIterator] = None
        self.smem_iter_B: Optional[GemmSmemIterator] = None

        self.smem_A_ptr: Optional[ArrayPtr] = None
        self.smem_B_ptr: Optional[ArrayPtr] = None

        self.support_optimize = False
        if op_type == ConvOpType.kForward and DepthwiseWarpIterArrangement.satisfied(spec):
            self.support_optimize = True
            self.depthwise_warp_arrangement = DepthwiseWarpIterArrangement(spec)
            self.add_param_class("DepthwiseWarpIterArrangement", self.depthwise_warp_arrangement, "DepthwiseWarpIter")
            self.add_member("depth_warp_iter", "DepthwiseWarpIter")
        

        self.add_code_before_class("""
        template <int a, int b>
        struct SimpleMax{
            static const int value = (a > b) ? a:b;
        };
        template <typename T1, typename T2>
        struct SimpleUnion{
            static const int size = SimpleMax<sizeof(T1), sizeof(T2)> :: value;
            
            TV_DEVICE_INLINE
            T1& first(){
                return *(reinterpret_cast<T1*>(storage));
            }
            TV_DEVICE_INLINE
            T2& second(){
                return *(reinterpret_cast<T2*>(storage));
            }
            
            private:
            char storage[size];

        };
        
        """)

    def min_arch(self) -> Optional[Tuple[int, int]]:
        return self.wmma.min_arch()
    
    @pccm.cuda.constructor(device=True, forceinline=True)   
    def ctor(self):
        code = Mma.ctor(self)
        if self.support_optimize:
            code.ctor_init("depth_warp_iter", "lane_idx")
        
        code.raw(f"""
            /*
            warp_shape is {self.spec.warp_shape}
            thread_mma_shape is {self.spec.thread_mma_shape}
            lane_mma_shape is {self.spec.lane_mma_shape}
            qw
            */
        """)

        return code
    #     code = pccm.code()
    #     code.arg("smem_storage", "GemmStorage*")
    #     code.arg("thread_idx,warp_idx_k,warp_m,warp_n,lane_idx", "int")
    #     code.ctor_init(
    #         "warp_iter_A",
    #         "smem_storage->smem_A.data(), warp_idx_k, warp_m, lane_idx")
    #     code.ctor_init(
    #         "warp_iter_B",
    #         "smem_storage->smem_B.data(), warp_idx_k, warp_n, lane_idx")
    #     code.ctor_init(
    #         "smem_iter_A",
    #         f"{self.smem_storage.smem_shape_a[1]}, smem_storage->smem_A.data(), thread_idx"
    #     )
    #     code.ctor_init(
    #         "smem_iter_B",
    #         f"{self.smem_storage.smem_shape_b[1]}, smem_storage->smem_B.data(), thread_idx"
    #     )
    #     return code

    def call_mask_sparse_k_first_fprop_depthwise(self):
        '''
        from pipelined mma.
        only load 1 time iteratorA, also inc_k 1 time
        but smem_iterA write many times
        '''
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        code.arg("mask", f"uint32_t")
        code.arg("RS", f"int")
        code.raw(f"""
        accumulators = src_accumulators;
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        """)
        # for inc filter first, 0123 0123 0123 0123
        code.raw(f"""
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = 1;
        // mask = 0xffffffff;
        MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
        // find initial gemm index
        int channel_start_idx = 0;
        while (!mask_iter.valid()){{
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
        }}
        // now input iter point to a valid location, mask iter point to this location too.
        input_iter_A.update_indices();
        input_iter_A.load(input_frag_A);
        input_iter_B.load(input_frag_B);
        
        input_iter_A.increment_k();
        input_iter_B.increment_k();
        
        smem_iter_A.store(input_frag_A);
        smem_iter_B.store(input_frag_B);
        __syncthreads();
        ++smem_iter_A;
        ++smem_iter_B;

        warp_iter_A.set_kgroup_index(channel_start_idx);
        warp_iter_B.set_kgroup_index(channel_start_idx);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        ++warp_iter_A;
        ++warp_iter_B;

        while (!mask_iter.end){{
            // tv::printf2_once(mask_iter.k_idx, mask_iter.filter_idx);
            for (int i = 0; i < gemm_k_iterations; ++i){{
                TV_PRAGMA_UNROLL
                for (int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                    if (warp_mma_k == {self.spec.num_warp_mma_iters} - 1) {{
                        // save to S1
                        smem_iter_A.store(input_frag_A);
                        smem_iter_B.store(input_frag_B);
                        __syncthreads();
                        ++smem_iter_A;
                        ++smem_iter_B;
                        // SMEM double buffer
                        if (smem_write_stage_idx == 1) {{
                            // back to S0
                            smem_iter_A.tile_increment(-{self.num_stage});
                            smem_iter_B.tile_increment(-{self.num_stage});
                        }} else {{
                            // 
                            warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                            warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                        }}
                        smem_write_stage_idx ^= 1;
                        if (i == gemm_k_iterations - 1)
                            channel_start_idx = 0;
                        else
                            channel_start_idx += {self.spec.num_warp_mma_iters}; 
                    }}
                    warp_iter_A.set_kgroup_index(channel_start_idx + (warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_B.set_kgroup_index(channel_start_idx + (warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                    warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                    ++warp_iter_A;
                    ++warp_iter_B;
                    // load next input frag
                    // to hide long input latency
                    // by a whole wmma operation
                    if (warp_mma_k == 0){{
                        // 01 001
                        // here input iter point to next location of current
                        // mask iter (may be invalid), we need to increment to
                        // find a valid location.
                        if (i == gemm_k_iterations - 1){{

                            input_iter_A.reset_k();
                            input_iter_B.reset_k();

                            ++mask_iter;

                            input_iter_A.increment_filter();
                            input_iter_B.increment_filter();

                            while (!mask_iter.valid() && !mask_iter.end){{
                                ++mask_iter;
                                input_iter_A.increment_filter();
                                input_iter_B.increment_filter();
                            }}
                            // load next indices
                            // TODO why do we need 20 more registers when use if?
                            input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_A.update_indices();
                            input_iter_B.load(input_frag_B);
                            input_iter_B.increment_k();
                        }}
                        input_iter_A.load(input_frag_A);
                        input_iter_A.increment_k();
                    }}
                    warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                            warp_frag_B[warp_mma_k % 2], accumulators);
                }}
            }}
        }}
        """)
        return code

    
    def call_mask_sparse_k_first_fprop_depthwise_V2(self):
        '''
        from pipelined mma.
        only load 1 time iteratorA, also inc_k 1 time
        but for smem iter B: only use one stage to save
        '''
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        code.arg("mask", f"uint32_t")
        code.arg("RS", f"int")
        code.raw(f"""
        accumulators = src_accumulators;
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        """)
        # for inc filter first, 0123 0123 0123 0123
        code.raw(f"""
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = 1, smemb_write_stage_idx = 1;
        // mask = 0xffffffff;
        MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
        // find initial gemm index
        int channel_start_idx = 0;
        while (!mask_iter.valid()){{
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
        }}
        // now input iter point to a valid location, mask iter point to this location too.
        input_iter_A.update_indices();
        input_iter_A.load(input_frag_A);
        input_iter_B.load(input_frag_B);
        
        input_iter_A.increment_k();
        input_iter_B.increment_k();
        
        smem_iter_A.store(input_frag_A);
        smem_iter_B.store(input_frag_B);
        __syncthreads();
        ++smem_iter_A;
        ++smem_iter_B;

        warp_iter_A.set_kgroup_index(channel_start_idx);
        warp_iter_B.set_kgroup_index(channel_start_idx);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        ++warp_iter_A;
        ++warp_iter_B;

        while (!mask_iter.end){{
            // tv::printf2_once(mask_iter.k_idx, mask_iter.filter_idx);
            for (int i = 0; i < gemm_k_iterations; ++i){{
                TV_PRAGMA_UNROLL
                for (int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                    if (warp_mma_k == {self.spec.num_warp_mma_iters} - 1) {{
                        // save to S1
                        smem_iter_A.store(input_frag_A);
                        if (i == gemm_k_iterations - 1){{
                            smem_iter_B.store(input_frag_B);
                            ++smem_iter_B;
                            if (smemb_write_stage_idx){{
                                smem_iter_B.tile_increment(-{self.num_stage});
                                warp_iter_B.tile_increment({self.partk} * {self.spec.num_warp_mma_iters});
                            }} else {{
                                warp_iter_B.tile_increment(- {self.partk} * {self.spec.num_warp_mma_iters});
                            }}
                            smemb_write_stage_idx ^= 1;
                        }}
                        __syncthreads();
                        ++smem_iter_A;
                        // SMEM double buffer
                        if (smem_write_stage_idx == 1) {{
                            // back to S0
                            smem_iter_A.tile_increment(-{self.num_stage});
                        }} else {{
                            // 
                            warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                        }}
                        warp_iter_B.tile_increment(-{self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        smem_write_stage_idx ^= 1;
                        if (i == gemm_k_iterations - 1)
                            channel_start_idx = 0;
                        else
                            channel_start_idx += {self.spec.num_warp_mma_iters}; 
                    }}
                    warp_iter_A.set_kgroup_index(channel_start_idx + (warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_B.set_kgroup_index(channel_start_idx + (warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                    warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                    ++warp_iter_A;
                    ++warp_iter_B;
                    // load next input frag
                    // to hide long input latency
                    // by a whole wmma operation
                    if (warp_mma_k == 0){{
                        // 01 001
                        // here input iter point to next location of current
                        // mask iter (may be invalid), we need to increment to
                        // find a valid location.
                        if (i == gemm_k_iterations - 1){{

                            input_iter_A.reset_k();
                            input_iter_B.reset_k();

                            ++mask_iter;

                            input_iter_A.increment_filter();
                            input_iter_B.increment_filter();

                            while (!mask_iter.valid() && !mask_iter.end){{
                                ++mask_iter;
                                input_iter_A.increment_filter();
                                input_iter_B.increment_filter();
                            }}
                            input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_A.update_indices();
                            input_iter_B.load(input_frag_B);
                            input_iter_B.increment_k();
                        }}
                        input_iter_A.load(input_frag_A);
                        input_iter_A.increment_k();
                    }}
                    warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                            warp_frag_B[warp_mma_k % 2], accumulators);
                }}
            }}
        }}
        """)
        return code
    
    def call_mask_sparse_k_first_fprop_depthwise_Optim(self):
        '''
        from pipelined mma.
        only load 1 time iteratorA, also inc_k 1 time
        but for smem iter B: only use one stage to save
        '''
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        code.arg("mask", f"uint32_t")
        code.arg("RS", f"int")
        code.raw(f"""
        accumulators = src_accumulators;
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        """)
        # for inc filter first, 0123 0123 0123 0123
        code.raw(f"""
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        int warp_frag_valid_b[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = 1, smemb_write_stage_idx = 1;
        // mask = 0xffffffff;
        MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
        // find initial gemm index
        int channel_start_idx = 0;
        while (!mask_iter.valid()){{
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
        }}
        // now input iter point to a valid location, mask iter point to this location too.
        input_iter_A.update_indices();
        input_iter_A.load(input_frag_A);
        input_iter_B.load(input_frag_B);
        
        input_iter_A.increment_k();
        input_iter_B.increment_k();
        
        smem_iter_A.store(input_frag_A);
        smem_iter_B.store(input_frag_B);
        __syncthreads();
        ++smem_iter_A;
        ++smem_iter_B;

        depth_warp_iter.increment_and_calc(warp_iter_A, warp_iter_B, warp_frag_valid_b[0]);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        while (!mask_iter.end){{
            // tv::printf2_once(mask_iter.k_idx, mask_iter.filter_idx);
            for (int i = 0; i < {self.depthwise_warp_arrangement.gemm_k_iters}; ++i){{
                TV_PRAGMA_UNROLL
                for (int warp_mma_k = 0; warp_mma_k < {self.depthwise_warp_arrangement.warp_mma_iters}; ++warp_mma_k){{
                    if (warp_mma_k == {self.depthwise_warp_arrangement.warp_mma_iters} - 1) {{
                        // save to S1
                        smem_iter_A.store(input_frag_A);
                        if (i == gemm_k_iterations - 1){{
                            smem_iter_B.store(input_frag_B);
                            ++smem_iter_B;
                            if (smemb_write_stage_idx){{
                                smem_iter_B.tile_increment(-{self.num_stage});
                                warp_iter_B.tile_increment({self.partk} * {self.spec.num_warp_mma_iters});
                            }} else {{
                                warp_iter_B.tile_increment(- {self.partk} * {self.spec.num_warp_mma_iters});
                            }}
                            smemb_write_stage_idx ^= 1;
                        }}
                        __syncthreads();
                        ++smem_iter_A;
                        // SMEM double buffer
                        if (smem_write_stage_idx == 1) {{
                            // back to S0
                            smem_iter_A.tile_increment(-{self.num_stage});
                        }} else {{
                            // 
                            warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                        }}
                        warp_iter_B.tile_increment(-{self.partk} *
                                                {self.spec.num_warp_mma_iters});
                        smem_write_stage_idx ^= 1;
                        if (i == gemm_k_iterations - 1)
                            channel_start_idx = 0;
                        else
                            channel_start_idx += {self.spec.num_warp_mma_iters}; 
                    }}

                    
                    depth_warp_iter.increment_and_calc(warp_iter_A, warp_iter_B, warp_frag_valid_b[(warp_mma_k + 1) % 2], i, warp_mma_k);
                    warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                    warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                    // load next input frag
                    // to hide long input latency
                    // by a whole wmma operation
                    if (warp_mma_k == 0){{
                        // 01 001
                        // here input iter point to next location of current
                        // mask iter (may be invalid), we need to increment to
                        // find a valid location.
                        if (i == gemm_k_iterations - 1){{

                            input_iter_A.reset_k();
                            input_iter_B.reset_k();

                            ++mask_iter;

                            input_iter_A.increment_filter();
                            input_iter_B.increment_filter();

                            while (!mask_iter.valid() && !mask_iter.end){{
                                ++mask_iter;
                                input_iter_A.increment_filter();
                                input_iter_B.increment_filter();
                            }}
                            input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_A.update_indices();
                            input_iter_B.load(input_frag_B);
                            input_iter_B.increment_k();
                        }}
                        input_iter_A.load(input_frag_A);
                        input_iter_A.increment_k();
                    }}
                    warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                            warp_frag_B[warp_mma_k % 2], accumulators); //, warp_frag_valid_b[warp_mma_k % 2]);
                    /*
                    for (int j=0; j<warp_frag_B[warp_mma_k % 2].size(); ++j)
                        if (j != warp_frag_valid_b[warp_mma_k % 2]){{
                            auto v = warp_frag_B[warp_mma_k % 2][j];
                            assert((float)v >= -1e-6 && (float)v <= 1e-6);
                        }}
                    */
                }}
            }}
        }}
        """)
        return code

    def call_mask_sparse_k_first_fprop_depthwise_Optim_1warp_iter(self):
        '''
        gemm_k_iter and warp_mma_iter = 1
        '''
        assert self.depthwise_warp_arrangement.gemm_k_iters == 1 and self.depthwise_warp_arrangement.warp_mma_iters == 1
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        code.arg("mask", f"uint32_t")
        code.arg("RS", f"int")
        code.raw(f"""
        accumulators = src_accumulators;
        SimpleUnion<{self.input_spec.input_iter_a.fragment_t}, {self.spec.warp_iter_a.fragment_t}> frag_A;
        SimpleUnion<{self.input_spec.input_iter_b.fragment_t}, {self.spec.warp_iter_b.fragment_t}> frag_B;
        
        int warp_frag_valid_b;
        WarpMma warp_mma;
        int smem_write_stage_idx = 1;
        // mask = 0xffffffff;
        MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
        // find initial gemm index

        while (!mask_iter.valid()){{
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
        }}
        // now input iter point to a valid location, mask iter point to this location too.
        input_iter_A.update_indices();
        input_iter_A.load(frag_A.first());
        input_iter_B.load(frag_B.first());
        
        input_iter_A.increment_k();
        input_iter_B.increment_k();
        
        smem_iter_A.store(frag_A.first());
        // smem_iter_B.store(frag_B.first());
        __syncthreads();
        ++smem_iter_A;
        // m++smem_iter_B;

        //depth_warp_iter.increment_and_calc(warp_iter_A, warp_iter_B, warp_frag_valid_b);
        depth_warp_iter(warp_iter_A);

        warp_iter_A.load(frag_A.second());
        // warp_iter_B.load_depthwise_fprop_no_mask(frag_B.second(), 0);

        while (!mask_iter.end){{
            // tv::printf2_once(mask_iter.k_idx, mask_iter.filter_idx);


            warp_mma(accumulators, frag_A.second(),
                frag_B.second(), accumulators);

            input_iter_A.reset_k();
            input_iter_B.reset_k();

            ++mask_iter;

            input_iter_A.increment_filter();
            input_iter_B.increment_filter();

            while (!mask_iter.valid() && !mask_iter.end){{
                ++mask_iter;
                input_iter_A.increment_filter();
                input_iter_B.increment_filter();
            }}
            input_iter_A.clear_all_mask_if_pred(mask_iter.end);
            input_iter_B.clear_all_mask_if_pred(mask_iter.end);
            input_iter_A.update_indices();
            input_iter_A.load(frag_A.first());
            input_iter_B.load(frag_B.first());
            input_iter_A.increment_k();
            input_iter_B.increment_k();

            
            smem_iter_A.store(frag_A.first());
            // smem_iter_B.store(frag_B.first());
            __syncthreads();
            // ++smem_iter_B;
            ++smem_iter_A;

            if (smem_write_stage_idx == 1) {{
                // back to S0
                smem_iter_A.tile_increment(-{self.num_stage});
                // smem_iter_B.tile_increment(-{self.num_stage});
                // warp_iter_B.tile_increment({self.partk} * {self.spec.num_warp_mma_iters});
            }} else {{
                // 
                warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                        {self.spec.num_warp_mma_iters});
                // warp_iter_B.tile_increment(- {self.partk} * {self.spec.num_warp_mma_iters});
            }}
            // warp_iter_B.tile_increment(-{self.partk} *
            //                         {self.spec.num_warp_mma_iters});
            smem_write_stage_idx ^= 1;

            // depth_warp_iter.increment_and_calc(warp_iter_A, warp_iter_B, warp_frag_valid_b, 0, 0);
            depth_warp_iter(warp_iter_A, 0, 0);
            warp_iter_A.load(frag_A.second());
            // warp_iter_B.load_depthwise_fprop_no_mask(frag_B.second(), 0);

        }}
        """)
        return code
    

    def call_mask_sparse_k_first_bwdI_depthwise(self):
        '''
        from pipelined mma.
        k_group_idx modified.
        '''
        code = pccm.code()
        code.arg("gemm_k_iterations", f"int")
        code.arg("accumulators", f"{self.accumulator_fragment}&")
        code.arg("input_iter_A", f"InputIteratorA &")
        code.arg("input_iter_B", f"InputIteratorB &")
        code.arg("src_accumulators", f"{self.accumulator_fragment} const&")
        code.arg("mask", f"uint32_t")
        code.arg("RS", f"int")
        code.raw(f"""
        accumulators = src_accumulators;
        {self.input_spec.input_iter_a.fragment_t} input_frag_A;
        {self.input_spec.input_iter_b.fragment_t} input_frag_B;
        """)
        # for inc filter first, 0123 0123 0123 0123
        code.raw(f"""
        {self.spec.warp_iter_a.fragment_t} warp_frag_A[2];
        {self.spec.warp_iter_b.fragment_t} warp_frag_B[2];
        WarpMma warp_mma;
        int smem_write_stage_idx = 1;
        // mask = 0xffffffff;
        MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);
        // find initial gemm index
        int channel_start_idx = 0;
        while (!mask_iter.valid()){{
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
        }}
        // now input iter point to a valid location, mask iter point to this location too.
        input_iter_A.update_indices();
        input_iter_A.load(input_frag_A);
        input_iter_B.load(input_frag_B);
        
        input_iter_A.increment_k();
        input_iter_B.increment_k();
        
        smem_iter_A.store(input_frag_A);
        smem_iter_B.store(input_frag_B);
        __syncthreads();
        ++smem_iter_A;
        ++smem_iter_B;

        warp_iter_A.set_kgroup_index(channel_start_idx);
        warp_iter_B.set_kgroup_index(channel_start_idx);

        warp_iter_A.load(warp_frag_A[0]);
        warp_iter_B.load(warp_frag_B[0]);

        ++warp_iter_A;
        ++warp_iter_B;

        while (!mask_iter.end){{
            // tv::printf2_once(mask_iter.k_idx, mask_iter.filter_idx);
            for (int i = 0; i < gemm_k_iterations; ++i){{
                TV_PRAGMA_UNROLL
                for (int warp_mma_k = 0; warp_mma_k < {self.spec.num_warp_mma_iters}; ++warp_mma_k){{
                    if (warp_mma_k == {self.spec.num_warp_mma_iters} - 1) {{
                        // save to S1
                        smem_iter_A.store(input_frag_A);
                        smem_iter_B.store(input_frag_B);
                        __syncthreads();
                        ++smem_iter_A;
                        ++smem_iter_B;
                        // SMEM double buffer
                        if (smem_write_stage_idx == 1) {{
                            // back to S0
                            smem_iter_A.tile_increment(-{self.num_stage});
                            smem_iter_B.tile_increment(-{self.num_stage});
                        }} else {{
                            // 
                            warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                            warp_iter_B.tile_increment(-{self.num_stage * self.partk} *
                                                    {self.spec.num_warp_mma_iters});
                        }}
                        smem_write_stage_idx ^= 1;
                        if (i == gemm_k_iterations - 1)
                            channel_start_idx = 0;
                        else
                            channel_start_idx += {self.spec.num_warp_mma_iters}; 
                    }}
                    warp_iter_A.set_kgroup_index(channel_start_idx + (warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_B.set_kgroup_index(channel_start_idx + (warp_mma_k + 1) % {self.spec.num_warp_mma_iters});
                    warp_iter_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
                    warp_iter_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);

                    ++warp_iter_A;
                    ++warp_iter_B;
                    // load next input frag
                    // to hide long input latency
                    // by a whole wmma operation
                    if (warp_mma_k == 0){{
                        // 01 001
                        // here input iter point to next location of current
                        // mask iter (may be invalid), we need to increment to
                        // find a valid location.
                        if (i == gemm_k_iterations - 1){{

                            input_iter_A.reset_k();
                            input_iter_B.reset_k();

                            ++mask_iter;

                            input_iter_A.increment_filter();
                            input_iter_B.increment_filter();

                            while (!mask_iter.valid() && !mask_iter.end){{
                                ++mask_iter;
                                input_iter_A.increment_filter();
                                input_iter_B.increment_filter();
                            }}
                            // load next indices
                            // TODO why do we need 20 more registers when use if?
                            input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                            input_iter_A.update_indices();
                        }}
                        input_iter_B.load(input_frag_B);
                        input_iter_A.load(input_frag_A);
                        input_iter_B.increment_k();
                        input_iter_A.increment_k();
                    }}
                    warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                            warp_frag_B[warp_mma_k % 2], accumulators);
                }}
            }}
        }}
        """)
        return code


    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call(self):
        if self.op_type == ConvOpType.kForward:
            if self.support_optimize:
                if self.depthwise_warp_arrangement.gemm_k_iters == 1 and self.depthwise_warp_arrangement.warp_mma_iters == 1:
                    return self.call_mask_sparse_k_first_fprop_depthwise_Optim_1warp_iter()
                else:
                    return self.call_mask_sparse_k_first_fprop_depthwise_Optim()
            else:
                return self.call_mask_sparse_k_first_fprop_depthwise_V2()
        elif self.op_type == ConvOpType.kBackwardInput:
            return self.call_mask_sparse_k_first_bwdI_depthwise()
        else:
            return Mma.call_mask_sparse_wgrad(self)
    
