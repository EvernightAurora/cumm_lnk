# Copyright 2022
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

from typing import List, Tuple, Union

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.common import GemmBasic, TensorViewNVRTC
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm import bases, constants, core, layout, thread_map
from cumm.gemm.arch import instmma


class WarpMmaSimtDepthwise(bases.WarpMma):
    def __init__(self, thread_mma_shape: Tuple[int, int,
                                               int], dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType, dtype_c: dtypes.DType, trans_a: bool,
                 trans_b: bool, trans_c: bool):
        # TODO merge mma sync
        super().__init__()
        self.add_dependency(TensorViewNVRTC, layout.RowMajor,
                            layout.ColumnMajor)
        self.thread_mma_shape = (thread_mma_shape[0], thread_mma_shape[1],
                                 thread_mma_shape[2])
        self.dtype_a = dtype_a
        self.dtype_b = dtype_b
        self.dtype_c = dtype_c
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.mn = thread_mma_shape[0] * thread_mma_shape[1]

        self.fragment_a_t = self.array_type(str(dtype_a), self.mn)
        self.fragment_b_t = self.array_type(str(dtype_b), self.thread_mma_shape[1])
        self.fragment_c_t = self.array_type(str(dtype_c), self.mn)
        self.mma = instmma.InstMma((1, 1, 1), 1, dtype_a, dtype_b, dtype_c,
                                       trans_a, trans_b, trans_c)
        self.add_param_class("instmma", self.mma, "InstMma")

    def array_type(self, dtype: Union[str, dtypes.DType], count: int):
        return core.array_type(dtype, count)

    def python_ctor(self):
        return self

    @pccm.cuda.member_function(name="operator()",
                               device=True,
                               forceinline=True)
    def call_operator(self):

        code = pccm.code()
        code.arg("D", f"{self.fragment_c_t}&")
        code.arg("A", f"{self.fragment_a_t} const &")
        code.arg("B", f"{self.fragment_b_t} const &")
        code.arg("C", f"{self.fragment_c_t} const &")


        code.raw(f"""
        InstMma mma;
        D = C;
        TV_PRAGMA_UNROLL
        for (int n = 0; n < {self.thread_mma_shape[1]}; ++n) {{
            TV_PRAGMA_UNROLL
            for (int m = 0; m < {self.thread_mma_shape[0]}; ++m) {{

                int m_serpentine = (n % 2) ? ({self.thread_mma_shape[0]} - 1 - m) : m;
                {self.mma.fragment_c_t} d;
                {self.mma.fragment_a_t} a;
                {self.mma.fragment_b_t} b;
                d[0] = D[m_serpentine * {self.thread_mma_shape[1]} + n];
                a[0] = A[m_serpentine * {self.thread_mma_shape[1]} + n];
                b[0] = B[n];
                mma(d, a, b, d);
                D[m_serpentine * {self.thread_mma_shape[1]} + n] = d[0];
            }}
        }}

        """)
        return code
    
    @pccm.cuda.member_function(name="operator()",
                               device=True,
                               forceinline=True)
    def call_operator_one(self):

        code = pccm.code()
        code.arg("D", f"{self.fragment_c_t}&")
        code.arg("A", f"{self.fragment_a_t} const &")
        code.arg("B", f"{self.fragment_b_t} const &")
        code.arg("C", f"{self.fragment_c_t} const &")
        code.arg("m", "const int&")


        code.raw(f"""
        InstMma mma;
        D = C;
        TV_PRAGMA_UNROLL
        for (int n = 0; n < {self.thread_mma_shape[1]}; ++n) {{
            {self.mma.fragment_c_t} d;
            {self.mma.fragment_a_t} a;
            {self.mma.fragment_b_t} b;
            d[0] = D[m * {self.thread_mma_shape[1]} + n];
            a[0] = A[m * {self.thread_mma_shape[1]} + n];
            b[0] = B[n];
            mma(d, a, b, d);
            D[m * {self.thread_mma_shape[1]} + n] = d[0];
        }}

        """)
        return code
    

    async def __call__(self, D: ArrayPtr, A: ArrayPtr, B: ArrayPtr,
                       C: ArrayPtr):
        raise NotImplementedError
