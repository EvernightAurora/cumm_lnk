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

    async def python_ctor(self, smem_A_ptr: ArrayPtr, smem_B_ptr: ArrayPtr,
                          thread_idx: int, warp_idx_k: int, warp_m: int,
                          warp_n: int, lane_idx: int):
        new_obj = MmaDepthwiseConvPipelined(self.dtype_acc, self.partk, self.num_stage, self.spec,
                      self.smem_storage, self.first_input_clear,
                      self.clear_mask, self.mask_sparse, self.increment_k_first,
                      self.op_type)
        new_obj.warp_iter_A = await self.spec.warp_iter_a.python_ctor(
            smem_A_ptr, warp_idx_k, warp_m, lane_idx)
        new_obj.warp_iter_B = await self.spec.warp_iter_b.python_ctor(
            smem_B_ptr, warp_idx_k, warp_n, lane_idx)
        new_obj.smem_iter_A = self.spec.smem_iter_a.python_ctor(
            self.smem_storage.smem_shape_a[1], smem_A_ptr, thread_idx)
        new_obj.smem_iter_B = self.spec.smem_iter_b.python_ctor(
            self.smem_storage.smem_shape_b[1], smem_B_ptr, thread_idx)
        new_obj.smem_A_ptr = smem_A_ptr
        new_obj.smem_B_ptr = smem_B_ptr
        if self.support_optimize:
            new_obj.depthwise_warp_iter = self.depthwise_warp_arrangement.python_ctor(lane_idx)
        return new_obj

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
    
    def call_mask_sparse_k_first_1warp_real_pipelined(self):
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
        SimpleUnion<{self.input_spec.input_iter_a.fragment_t}, {self.spec.warp_iter_a.fragment_t}> frag_A[2];


        SimpleUnion<{self.input_spec.input_iter_b.fragment_t}, {self.spec.warp_iter_b.fragment_t}> frag_B[2];
        
        int warp_frag_valid_b;
        WarpMma warp_mma;
        int smem_write_stage_idx = 1;
        MaskIGemmIterator mask_iter(gemm_k_iterations, RS, mask);

        while (!mask_iter.valid()){{
            ++mask_iter;
            input_iter_A.increment_filter();
            input_iter_B.increment_filter();
        }}

        input_iter_A.update_indices();
        input_iter_A.load(frag_A[0].first());
        input_iter_B.load(frag_B[0].first());
        
        input_iter_A.increment_k();
        input_iter_B.increment_k();
        
        smem_iter_A.store(frag_A[0].first());
        ++smem_iter_A;
        depth_warp_iter(warp_iter_A);

        

        while (!mask_iter.end){{
            TV_PRAGMA_UNROLL
            for (int stage_idx = 0; stage_idx < 2; ++ stage_idx){{

                input_iter_A.reset_k();
                input_iter_B.reset_k();
                input_iter_A.increment_filter();
                input_iter_B.increment_filter();

                ++mask_iter;
                while (!mask_iter.valid() && !mask_iter.end){{
                    ++mask_iter;
                    input_iter_A.increment_filter();
                    input_iter_B.increment_filter();
                }}
                input_iter_A.clear_all_mask_if_pred(mask_iter.end);
                input_iter_B.clear_all_mask_if_pred(mask_iter.end);
                input_iter_A.update_indices();
                input_iter_A.load(frag_A[stage_idx ^ 1].first());
                input_iter_B.load(frag_B[stage_idx ^ 1].first());

                __syncthreads();
                warp_iter_A.load(frag_A[stage_idx].second());
                
                input_iter_A.increment_k();
                input_iter_B.increment_k();

                warp_mma(accumulators, frag_A[stage_idx].second(),
                    frag_B[stage_idx].second(), accumulators);

                smem_iter_A.store(frag_A[stage_idx ^ 1].first());
                ++smem_iter_A;
                if (stage_idx == 0) {{
                    smem_iter_A.tile_increment(-{self.num_stage});
                }} else {{
                    warp_iter_A.tile_increment(-{self.num_stage * self.partk} *
                                            {self.spec.num_warp_mma_iters});
                }}

                depth_warp_iter(warp_iter_A, 0, 0);

            }}
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
                    return self.call_mask_sparse_k_first_1warp_real_pipelined()
                else:
                    return self.call_mask_sparse_k_first_fprop_depthwise_Optim()
            else:
                return self.call_mask_sparse_k_first_fprop_depthwise_V2()
        elif self.op_type == ConvOpType.kBackwardInput:
            return self.call_mask_sparse_k_first_bwdI_depthwise()
        else:
            return Mma.call_mask_sparse_wgrad(self)
        
    async def __call__(self, gemm_k_iterations: int, accumulators: ArrayPtr,
                       input_iter_A: GemmInputIterator,
                       input_iter_B: GemmInputIterator,
                       src_accumulators: ArrayPtr):
        if self.op_type == ConvOpType.kBackwardWeight:
            return Mma.__call__(self, gemm_k_iterations, accumulators,
                                    input_iter_A, input_iter_B,
                                    src_accumulators)
        assert self.warp_iter_A is not None
        assert self.warp_iter_B is not None
        assert self.smem_iter_A is not None
        assert self.smem_iter_B is not None
        smem_iter_A = self.smem_iter_A
        smem_iter_B = self.smem_iter_B
        warp_iter_A = self.warp_iter_A
        warp_iter_B = self.warp_iter_B

        input_frag_A = ArrayPtr(input_iter_A.dtype.tv_dtype,
                                input_iter_A.element_count)
        input_frag_B = ArrayPtr(input_iter_B.dtype.tv_dtype,
                                input_iter_B.element_count)
        channel_start_idx = 0

        input_frag_A.clear()
        input_frag_B.clear()
        inp_coords_A_list = []
        inp_coords_B_list = []
        smem_coords_A_list = []
        smem_coords_B_list = []
        warp_coords_A_list = []
        warp_coords_B_list = []
        warp_frag_A_list = []
        warp_frag_B_list = []
        warp_frag_A = [
            ArrayPtr(warp_iter_A.dtype.tv_dtype,
                     self.spec.warp_iter_a.element_count) for _ in range(2)
        ]
        warp_frag_B = [
            ArrayPtr(warp_iter_B.dtype.tv_dtype,
                     self.spec.warp_iter_b.element_count) for _ in range(2)
        ]

        inp_coors_A = input_iter_A.load_python(input_frag_A)
        # print(inp_coors_A)
        inp_coords_A_list.append(inp_coors_A)
        # if cudasim.threadIdx().x < 32:
        #     print(cudasim.threadIdx().x, inp_coors_A)
        inp_coors_B = input_iter_B.load_python(input_frag_B)
        inp_coords_B_list.append(inp_coors_B)
        if cudasim.debug_once():
            print("GEMM ITERATIONS", gemm_k_iterations)
            inpd = input_frag_A.data.numpy_view()
            print("FirstInputA",
                  cudasim.blockIdx().z, inpd.mean(), inpd.max(), inpd.min())
            inpd = input_frag_B.data.numpy_view()
            print("FirstInputB",
                  cudasim.blockIdx().z, inpd.mean(), inpd.max(), inpd.min())

        input_iter_A.increment_python()
        input_iter_B.increment_python()
        smem_coords_A = await smem_iter_A.store_python(input_frag_A)
        smem_coords_B = await smem_iter_B.store_python(input_frag_B)
        smem_coords_A_list.append(smem_coords_A)
        smem_coords_B_list.append(smem_coords_B)

        await cudasim.syncthreads()
        # if cudasim.threadIdx().x == 0:
        #     smem_A = self.smem_A_ptr.data.numpy()
        #     print(smem_A.mean(), smem_A.min(), smem_A.max())
        #     print(input_frag_A.meta_data.numpy_view())
        #     print(smem_A_ptr.meta_data.numpy_view().astype(np.int32).reshape(-1, 128 + self.algo_spec.padding_mn[0])[:8, :10])

        smem_iter_A.increment_python()
        smem_iter_B.increment_python()

        warp_iter_A.set_wmma_k_index_python(channel_start_idx)
        warp_iter_B.set_wmma_k_index_python(channel_start_idx)

        warp_coords_A = await warp_iter_A.load_python(warp_frag_A[0])
        warp_coords_B = await warp_iter_B.load_python(warp_frag_B[0])
        # if (cudasim.threadIdx().x == 0):

        warp_coords_A_list.append(warp_coords_A)
        warp_coords_B_list.append(warp_coords_B)
        warp_frag_A_list.append(warp_frag_A[0].meta_data.numpy_view().copy())
        warp_frag_B_list.append(warp_frag_B[0].meta_data.numpy_view().copy())

        # if cudasim.threadIdx().x == 0:
        #     print(warp_frag_A[0].data.numpy_view().astype(np.int32))
        #     print(warp_frag_A[0].data.numpy_view().mean(), "WARP_A_FIRST")
        #     print(warp_frag_B[0].data.numpy_view().mean(), "WARP_B_FIRST")
        # if cudasim.threadIdx().x == 0:

        # print(cudasim.threadIdx().x, input_frag_A.data.mean(), "input_frag_A FIRST")
        # print(input_frag_B.data.mean(), "input_frag_B FIRST")
        warp_iter_A.increment_python()
        warp_iter_B.increment_python()
        if cudasim.debug_once():
            inpd = warp_frag_A[0].data.numpy_view()
            print("FirstWarpA",
                  cudasim.blockIdx().z, inpd.mean(), inpd.max(), inpd.min())
            print(inpd)
            inpd = warp_frag_B[0].data.numpy_view()
            print("FirstWarpB",
                  cudasim.blockIdx().z, inpd.mean(), inpd.max(), inpd.min())

        warp_mma = self.spec.warp_mma.python_ctor()
        smem_write_stage_idx = 1
        gemm_k_iterations_bkp = gemm_k_iterations
        if gemm_k_iterations <= 1:
            input_iter_A.clear_mask_python()
            input_iter_B.clear_mask_python()
        if cudasim.debug_once():
            print("gemm_k_iterations", gemm_k_iterations)
        while gemm_k_iterations > 0:
            for warp_mma_k in range(self.spec.num_warp_mma_iters):
                if (warp_mma_k == self.spec.num_warp_mma_iters - 1):
                    smem_coords_A = await smem_iter_A.store_python(input_frag_A
                                                                   )
                    smem_coords_B = await smem_iter_B.store_python(input_frag_B
                                                                   )
                    if len(smem_coords_A_list) < gemm_k_iterations_bkp:
                        smem_coords_A_list.append(smem_coords_A)
                        smem_coords_B_list.append(smem_coords_B)
                    await cudasim.syncthreads()
                    smem_iter_A.increment_python()
                    smem_iter_B.increment_python()

                    if (smem_write_stage_idx == 1):
                        smem_iter_A.tile_increment_python(-self.num_stage)
                        smem_iter_B.tile_increment_python(-self.num_stage)
                    else:
                        warp_iter_A.tile_increment_python(
                            -self.num_stage * self.partk *
                            self.spec.num_warp_mma_iters)
                        warp_iter_B.tile_increment_python(
                            -self.num_stage * self.partk *
                            self.spec.num_warp_mma_iters)

                    smem_write_stage_idx ^= 1
                    if gemm_k_iterations == 1:
                        channel_start_idx = 0
                    else:
                        channel_start_idx += self.spec.num_warp_mma_iters
                # if cudasim.threadIdx().x == 255:
                #     print(warp_mma_k)
                warp_iter_A.set_wmma_k_index_python(
                    (warp_mma_k + 1) % self.spec.num_warp_mma_iters)
                warp_iter_B.set_wmma_k_index_python(
                    (warp_mma_k + 1) % self.spec.num_warp_mma_iters)

                warp_coords_A = await warp_iter_A.load_python(
                    warp_frag_A[(warp_mma_k + 1) % 2])
                warp_coords_B = await warp_iter_B.load_python(
                    warp_frag_B[(warp_mma_k + 1) % 2])
                if len(
                        warp_frag_A_list
                ) != self.spec.num_warp_mma_iters * gemm_k_iterations_bkp:
                    warp_frag_A_list.append(
                        warp_frag_A[(warp_mma_k + 1) %
                                    2].meta_data.numpy_view().copy())
                    warp_frag_B_list.append(
                        warp_frag_B[(warp_mma_k + 1) %
                                    2].meta_data.numpy_view().copy())
                    warp_coords_A_list.append(warp_coords_A)
                    warp_coords_B_list.append(warp_coords_B)

                # if cudasim.threadIdx().x == 0:
                #     wa = warp_frag_A[(warp_mma_k + 1) % 2].data.numpy_view()
                #     print(wa.astype(np.int32))
                warp_iter_A.increment_python()
                warp_iter_B.increment_python()
                if (warp_mma_k == 0):
                    inp_coors_A = input_iter_A.load_python(input_frag_A)
                    inp_coors_B = input_iter_B.load_python(input_frag_B)
                    inp_coords_A_list.append(inp_coors_A)
                    inp_coords_B_list.append(inp_coors_B)
                    if cudasim.debug_once():
                        inpd = input_frag_A.data.numpy_view()
                        print("InputA",
                              cudasim.blockIdx().z, inpd.mean(), inpd.max(),
                              inpd.min())
                        inpd = input_frag_B.data.numpy_view()
                        print("InputB",
                              cudasim.blockIdx().z, inpd.mean(), inpd.max(),
                              inpd.min())

                    # if cudasim.threadIdx().x == 200:
                    #     print(input_frag_A.data.mean(), "INPUT A")
                    input_iter_A.increment_python()
                    input_iter_B.increment_python()
                    if (gemm_k_iterations <= 2):
                        input_iter_A.clear_mask_python()
                        input_iter_B.clear_mask_python()
                if cudasim.debug_once():
                    inpd = warp_frag_A[warp_mma_k % 2].data.numpy_view()
                    print(f"WarpA", warp_mma_k,
                          cudasim.blockIdx().z, inpd.mean(), inpd.max(),
                          inpd.min())
                    inpd = warp_frag_B[warp_mma_k % 2].data.numpy_view()
                    print(f"WarpB", warp_mma_k,
                          cudasim.blockIdx().z, inpd.mean(), inpd.max(),
                          inpd.min())
                await warp_mma(accumulators, warp_frag_A[warp_mma_k % 2],
                               warp_frag_B[warp_mma_k % 2], accumulators)

            gemm_k_iterations -= 1
        if cudasim.debug_once():
            acc = accumulators.data.numpy_view()
            cudasim.debug_print(
                f"accumulator {acc.mean()} , max: {acc.max()} , min: {acc.min()}"
            )

        res = {
            "InputA": {
                "input_coords": inp_coords_A_list,
                "smem_coords": smem_coords_A_list,
                "warp_coords": warp_coords_A_list,
                "smem_shape": smem_iter_A.get_smem_vis_shape(),
                "warp_frags": warp_frag_A_list,
                "input_epa": input_iter_A.element_per_acc,
                "smem_epa": smem_iter_A.element_per_acc,
                "warp_epa": warp_iter_A.element_per_acc,
            },
            "InputB": {
                "input_coords": inp_coords_B_list,
                "smem_coords": smem_coords_B_list,
                "warp_coords": warp_coords_B_list,
                "warp_frags": warp_frag_B_list,
                "smem_shape": smem_iter_B.get_smem_vis_shape(),
                "input_epa": input_iter_B.element_per_acc,
                "smem_epa": smem_iter_B.element_per_acc,
                "warp_epa": warp_iter_B.element_per_acc,
            },
        }
        return res


class MmaDepthwiseConvPipelinedV2(GemmComponentBase):  # optim: no smem, only for fwd & bwdI
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
        assert op_type != ConvOpType.kBackwardWeight
        # super().__init__(dtype_acc, partk, num_stage, spec, smem_storage, first_input_clear,
        #                     clear_mask, mask_sparse, increment_k_first, is_sparse_wgrad)
        super().__init__()

        self.dtype_acc = dtype_acc
        miter = MaskIGemmIterator(increment_k_first)
        self.add_param_class("mma_ns_miter", miter, "MaskIGemmIterator")

        self.smem_storage = smem_storage
        self.spec = spec
        self.num_stage = num_stage
        self.mask_sparse = mask_sparse
        self.increment_k_first = increment_k_first
        self.partk = partk
        self.first_input_clear = first_input_clear
        self.clear_mask = clear_mask
        self.input_spec = spec.input_spec
        self.add_param_class("mma_ns_gm", smem_storage, "GemmStorage")
        self.accumulator_fragment = array_type(dtype_acc,
                                               spec.accumulator_size)
        self.add_param_class("mma_ns_ia", self.input_spec.input_iter_a,
                             "InputIteratorA")
        self.add_param_class("mma_ns_ib", self.input_spec.input_iter_b,
                             "InputIteratorB")
        self.add_param_class("mma_ns_wmma", spec.warp_mma, "WarpMma")
        self.wmma = spec.warp_mma

        self.enable_v2 = False
        if self.spec.input_spec.input_iter_a.can_multistage_load():
            self.iterA_params = list(self.spec.input_spec.input_iter_a.enumurate_get_param())
            if len(self.iterA_params) != self.spec.thread_mma_shape[0]:
                return
            else:
                self.enable_v2 = True

        # cudasim

    def min_arch(self) -> Optional[Tuple[int, int]]:
        return self.wmma.min_arch()
    
    @pccm.cuda.constructor(device=True, forceinline=True)   
    def ctor(self):
        code = pccm.code()
        
        code.raw(f"""
            /*
            warp_shape is {self.spec.warp_shape}
            thread_mma_shape is {self.spec.thread_mma_shape}
            lane_mma_shape is {self.spec.lane_mma_shape}
            
            */
        """)
        code.arg("smem_storage", "void*")
        code.arg("thread_idx,warp_idx_k,warp_m,warp_n,lane_idx", "int")
        return code
    
    def call_mask_sparse_k_first_straight(self):
        '''
        from pipelined mma depthwise mma.
        no complex logical code.
        no smem used, only input iter

        for test,  this isn't pipeline
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

        while(!mask_iter.end){{
            input_iter_A.load(input_frag_A);
            input_iter_B.load(input_frag_B);
            
            input_iter_A.increment_k();
            input_iter_B.increment_k();

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
            input_iter_A.update_indices();

            warp_mma(accumulators, input_frag_A,
                            input_frag_B, accumulators);
        }}
        
        """)
        return code
    
    def call_mask_sparse_k_first_pipelined(self):
        '''
        from pipelined mma depthwise mma.
        no complex logical code.
        no smem used, only input iter
        pipelined
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
        {self.input_spec.input_iter_a.fragment_t} input_frag_A[2];
        {self.input_spec.input_iter_b.fragment_t} input_frag_B[2];
        
        WarpMma warp_mma;
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

        input_iter_A.load(input_frag_A[0]);
        input_iter_B.load(input_frag_B[0]);

        while(!mask_iter.end){{
            
            TV_PRAGMA_UNROLL
            for (int stage_idx = 0; stage_idx < 2; ++stage_idx){{
                input_iter_A.increment_k();
                input_iter_B.increment_k();

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

                input_iter_A.load(input_frag_A[stage_idx ^ 1]);
                input_iter_B.load(input_frag_B[stage_idx ^ 1]);

                warp_mma(accumulators, input_frag_A[stage_idx],
                            input_frag_B[stage_idx], accumulators);

            }}
        }}
        
        """)
        return code

    def call_mask_sparse_k_first_pipelined_V2(self):
        '''
        from pipelined mma depthwise mma.
        no complex logical code.
        no smem used, only input iter
        pipelined
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
        
        WarpMma warp_mma;
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

        while(!mask_iter.end){{
            
            TV_PRAGMA_UNROLL
            for (int warp_mma_subm = 0; warp_mma_subm < {len(self.iterA_params)}; ++warp_mma_subm){{
                if (warp_mma_subm == 0){{
                    input_iter_A.increment_k();
                    input_iter_B.increment_k();

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
                }}
                
                warp_mma(accumulators, input_frag_A,
                            input_frag_B, accumulators, warp_mma_subm);
                {"".join(
                    [f"if (warp_mma_subm == {i}) input_iter_A.load_ptr_with_param_to_frag({self.iterA_params[(i)%len(self.iterA_params)]}, input_frag_A);" for i in range(len(self.iterA_params))]
                )}
                
                

                if (warp_mma_subm == {len(self.iterA_params) - 1})
                    input_iter_B.load(input_frag_B);

            }}
        }}
        """)
        return code
    
    @pccm.cuda.member_function(device=True,
                               forceinline=True,
                               name="operator()")
    def call(self):
        if self.enable_v2:
            return self.call_mask_sparse_k_first_pipelined_V2()
        return self.call_mask_sparse_k_first_pipelined()