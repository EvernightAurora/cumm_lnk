# Copyright 2021 Fan Xie
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
from cumm.common import GemmBasic, TensorViewNVRTC
from cumm.gemm import bases, constants, core, layout, thread_map
from cumm.gemm.arch.memory import SharedLdSt


class DepthwiseWgradTramsformer(bases.GemmComponentBase):
    def __init__(self, mma_spec, trans_c, dtype_acc, tile_shape, warp_tile_shape, lane_layout) -> None:
        super().__init__()
        self.mma_spec = mma_spec
        self.add_dependency(TensorViewNVRTC, layout.RowMajor,
                            layout.ColumnMajor)
        self.lane_layout = lane_layout
        self.add_param_class("Lane_Layout", self.lane_layout, "LaneLayout")
        self.warp_tile_shape = warp_tile_shape
        self.tile_shape = tile_shape
        self.warp_shape = mma_spec.warp_shape       # 4, 8 or 8, 4
        self.thread_mma_shape = mma_spec.thread_mma_shape    # warp_tile_shape / warp_shape
        self.lane_mma_shape = mma_spec.lane_mma_shape       # per con
        assert mma_spec.lane_mma_shape[2] == 1, "Not Impl"
        self.trans_c = trans_c
        assert not self.trans_c, "Not impl"
        assert warp_tile_shape[0] == warp_tile_shape[1], "wm and wn must be equal"

        self.warp_count_shape = mma_spec.warp_count_shape
        self.warp_count = mma_spec.warp_count
        assert self.warp_shape == [4, 8] or self.warp_count == [8, 4], "Not Impl"
        assert self.tile_shape[0] == self.tile_shape[1], "depthwise wgrad only support same tile_m tile_n"

        # assert self.warp_count == 1, "Not Implement"

        self.fragment_c_t = core.array_type(dtype_acc, self.thread_mma_shape[0] * self.thread_mma_shape[1])

        self.dtype = dtype_acc
        self.SharedCp = SharedLdSt(self.dtype.itemsize())
        self.add_param_class("SharedLdSt", self.SharedCp, "SharedCp")
        transfer_type_dict = {
            1: "char",
            2: "int16_t",
            4: "int32_t",
            8: "int64_t"
        }
        assert dtype_acc.itemsize() in transfer_type_dict.keys(), "Weird dtype of " + str(dtype_acc)
        self.transfer_count = self.thread_mma_shape[0]
        self.add_member("layoutC", "ColumnMajor" if self.trans_c else "RowMajor")
        self.add_member("layoutW", "LaneLayout")
        self.add_member("w_m, w_n", "int")

    @pccm.cuda.constructor(device=True, forceinline=True) 
    def ctor(self):
        code = pccm.code()
        code.arg("lane_id", "int")
        L_C = "ColumnMajor" if self.trans_c else "RowMajor"
        code.ctor_init("layoutC", f"{L_C}::from_shape({{ {self.thread_mma_shape[0]}, {self.thread_mma_shape[1]} }})")
        code.ctor_init("layoutW", f"LaneLayout::from_shape({{{self.warp_shape[0]}, {self.warp_shape[1]}}})")
        code.raw(f"""
            w_m = layoutW.inverse_0(lane_id);
            w_n = layoutW.inverse_1(lane_id);
        """)
        return code
    
    def transfer_singlewarp(self):
        code = pccm.code()
        code.arg("frag", f"{self.fragment_c_t}&")
        code.raw("int seek_lane, save_idx, provide_idx;")
        code.raw(f"{self.dtype} get;")
        for i in range(self.transfer_count):
            '''
            warp(a, 0) seeking warp(a, i/thread_mma_shape[1] + a * (thread_mma_shape[0] / thread_mma_shape[1]))'s data
            it provide frag(i, i % thread_mma_shape[1]) element // if warpshape is (8, 4) is (i, i + (a%2)*tile_mma_shape[0])
            save to frag(i, 0)

            '''
            code.raw(f"""
                seek_lane = layoutW(w_m, {i // self.thread_mma_shape[1]} + w_m * {self.thread_mma_shape[0]} / {self.thread_mma_shape[1]});
                save_idx = layoutC({i}, 0);
            """)
            if self.warp_shape[0] > self.warp_shape[1]:
                code.raw(f"""
                provide_idx = layoutC({i}, {i} + (w_m % {self.warp_shape[0] / self.warp_shape[1]}) * {self.thread_mma_shape[0]});
                """)
            else:
                code.raw(f"""
                provide_idx = layoutC({i}, {i % self.thread_mma_shape[1]});
                """)
            code.raw(f"""
                get = __shfl_sync(0xFFFFFFFF, frag[provide_idx], seek_lane);
                frag[save_idx] = get;
            """)
        return code
    
    def transfer_singlewarp_correct(self, code):
        code.raw("int seek_lane, save_idx, provide_idx, m_infact, n_seek, n_at_warp_idx_n, n_in_fragment_idx_n;")
        code.raw(f"{self.dtype} get;")
        code.raw(f"""
            /*
            warp_shape is {self.warp_shape}
            thread_mma_shape is {self.thread_mma_shape}
            lane_mma_shape is {self.lane_mma_shape}
            qw
            */
        """)
        for i in range(self.transfer_count):
            lane_mma_m_idx = i // self.lane_mma_shape[0]
        
            code.raw(f"""
                m_infact = w_m * {self.lane_mma_shape[0]} + {lane_mma_m_idx * self.warp_shape[0] * self.lane_mma_shape[0]} + {i % self.lane_mma_shape[0]};
                n_seek = m_infact;
                n_at_warp_idx_n = (n_seek / {self.lane_mma_shape[1]}) % {self.warp_shape[1]};
                n_in_fragment_idx_n = (n_at_warp_idx_n / {self.warp_shape[1]}) * {self.lane_mma_shape[1]} + n_seek % {self.lane_mma_shape[1]};

                seek_lane = layoutW(w_m, n_at_warp_idx_n);
                provide_idx = layoutC({i}, n_in_fragment_idx_n);

                save_idx = layoutC({i}, 0);

            """)
            code.raw(f"""
                get = __shfl_sync(0xFFFFFFFF, frag[provide_idx], seek_lane);
                frag[save_idx] = get;

            """)
        return code
    
    def transfer_through_warp(self, code):
        """
            a tile: [T, T]
            move its diag:  at[i, i] for i in range
            to one line:    at[:, 0]
            
            shfl_sync can only move in one warp, if we want to through warp, we need smem
        """
        code.raw(f"""
            /*
            warp_shape is {self.warp_shape}
            thread_mma_shape is {self.thread_mma_shape}
            lane_mma_shape is {self.lane_mma_shape}
            qw
            */
        """)
        code.raw(f"""
            {self.dtype}* smem = reinterpret_cast<{self.dtype}*> (smem_workspace);
            int m_infact, n_seek, n_warp_idx, n_warp_residual, n_lane_idx_n, n_in_fragment_idx_n;
            int provide_idx, save_idx, operate_idx;
            bool store_valid, save_valid;
        """)
        for i in range(self.transfer_count):
            lane_mma_m_idx = i // self.lane_mma_shape[0]
        
            code.raw(f"""
                m_infact = warp_m * {self.warp_tile_shape[0]} + w_m * {self.lane_mma_shape[0]} + {lane_mma_m_idx * self.warp_shape[0] * self.lane_mma_shape[0]} + {i % self.lane_mma_shape[0]};
                n_seek = m_infact;
                n_warp_idx = n_seek / {self.warp_tile_shape[1]};
                n_warp_residual = n_seek % {self.warp_tile_shape[1]};
                n_lane_idx_n = (n_warp_residual / {self.lane_mma_shape[1]}) % {self.warp_shape[1]};
                n_in_fragment_idx_n = (n_lane_idx_n / {self.warp_shape[1]}) * {self.lane_mma_shape[1]} + n_warp_residual % {self.lane_mma_shape[1]};

                provide_idx = layoutC({i}, n_in_fragment_idx_n);
                save_idx = layoutC({i}, 0);
                store_valid = (n_warp_idx == warp_n && n_lane_idx_n == w_n);
                save_valid = (warp_n == 0 && w_n == 0);
                operate_idx = {i * (self.warp_tile_shape[0] // self.thread_mma_shape[0])} + w_m + warp_m * {self.warp_tile_shape[0]};
                operate_idx = operate_idx * {max(1, 4 / self.dtype.itemsize())};

                """)

            # code.raw(f"""
            #     if (blockIdx.z == 0 && blockIdx.y == 0){{
            #         __syncwarp();
            #         // printf("TF T%d,  {i}th p %d s %d ope %d\\n", threadIdx.x, provide_idx, save_idx, operate_idx);
            #         printf("TF T%d,  {i}th n %d nwarp %d n_lid %d lidx %d\\n", threadIdx.x, n_seek, n_warp_idx, n_lane_idx_n, n_in_fragment_idx_n);
                    
            #     }}
            #     __syncthreads();
            # """)
            code.raw("""
                SharedCp::store_pred(frag[provide_idx], (smem + operate_idx), store_valid);
                __syncthreads();
                SharedCp::load_pred(&frag[save_idx], (smem + operate_idx), save_valid);

                """)
            
        code.raw("__syncthreads();")
        return code



    @pccm.cuda.member_function(name="operator()",
                               device=True,
                               forceinline=True)
    def call(self):
        code = pccm.code()
        code.arg("frag", f"{self.fragment_c_t}&")
        code.arg("warp_m, warp_n", "int", "0")
        code.arg("smem_workspace", "void*", "nullptr")
        if self.warp_count == 1:
            return self.transfer_singlewarp_correct(code)
        else:
            return self.transfer_through_warp(code)






    pass
