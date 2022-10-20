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

from typing import List, Optional, Tuple

import numpy as np
import pccm

from cumm import cudasim, dtypes
from cumm.gemm import constants, core


class GlobalLoad(pccm.ParameterizedClass):
    def __init__(self,
                 load_bytes: int,
                 cache_op: Optional[pccm.cuda.CacheOpLd] = None,
                 prefetch: bool = False,
                 level: str = "",
                 prefetch_size: int = -1):
        super().__init__()
        self.load_bytes = load_bytes
        self.cache_op = cache_op
        self.load_dtype = dtypes.uint32
        self.prefetch = prefetch
        if prefetch_size != -1:
            assert prefetch_size in [64, 128, 256]
        if level:
            assert level == "L2"
        self.level = level
        self.prefetch_size = prefetch_size
        if load_bytes >= 4:
            self.count = self.load_bytes // 4
            assert self.load_bytes % 4 == 0
            self.fragment_t = core.array_type(str(self.load_dtype),
                                              self.load_bytes // 4)
        else:
            self.count = 1
            if load_bytes == 2:
                self.load_dtype = dtypes.uint16
            else:
                self.load_dtype = dtypes.uint8
            self.fragment_t = core.array_type(str(self.load_dtype), 1)

    def _run(self,
             code: pccm.cuda.PTXCode,
             level: str = "",
             prefetch_size: int = -1):
        with code.asm_block() as asm:
            ptr_addr = asm.global_ptr("ptr")
            frag_reg_type = pccm.cuda.RegDType.B32
            if self.load_dtype == dtypes.uint16:
                frag_reg_type = pccm.cuda.RegDType.U16
            frag = asm.reg_ptr("frag_ptr", frag_reg_type)
            pred = asm.ext_reg("(int)pred", pccm.cuda.RegDType.B32)
            for i in range(self.count):
                asm.mov(frag[i], frag[i])  # TODO WTF???

            with asm.pred_if("p", "ne", pred, 0) as reg:
                frag_unpack = frag.unpack(self.count)
                if self.count > 4:
                    num_vec_load = self.count // 4
                    for i in range(num_vec_load):
                        if self.prefetch:
                            asm.generic("prefetch.global.L2",
                                        [ptr_addr + i * 16])
                        asm.ld(ptr_addr + i * 16,
                               frag_unpack[i * 4:(i + 1) * 4], self.cache_op)
                else:
                    if self.prefetch:
                        asm.generic("prefetch.global.L2", [ptr_addr])
                    asm.ld(ptr_addr, frag_unpack, self.cache_op, level,
                           prefetch_size)

    @pccm.cuda.static_function(device=True, forceinline=True)
    def run(self):
        code = pccm.cuda.PTXCode()
        code.targ("Frag")
        code.arg("frag", f"Frag &")
        code.arg("ptr", "void const*")
        code.arg("pred", "bool")
        code.raw(
            f"{self.load_dtype}* frag_ptr = reinterpret_cast<{self.load_dtype}*>(&frag);"
        )
        if self.load_dtype == dtypes.uint8:
            code.raw(f"""
            if (pred){{
                reinterpret_cast<{self.load_dtype} const*>(ptr)[0] = frag_ptr[0];
            }}
            """)
        else:
            with code.macro_if_("CUDA_VERSION >= 11040"):
                self._run(code, self.level, self.prefetch_size)
            with code.macro_else_():
                self._run(code)
            code.macro_endif_()
        return code


class SharedLdSt(pccm.ParameterizedClass):
    def __init__(self, itemsize) -> None:
        super().__init__()
        self.itemsize = itemsize
        assert self.itemsize in [2, 4, 8, 16], "Not Support"
        self.add_include("tensorview/gemm/arch/memory_sm75.h")
        self.size_to_dtype = {
            2: "uint16_t",
            4: "uint32_t",
            8: "uint2",
            16: "uint4"
        }
        self.size_to_args = {
            2: 1, 4: 1,
            8: 2, 16: 4
        }
    
    def shared_load(self, code, pred=False):
        
        dtype_c = self.size_to_dtype[self.itemsize]
        args_c = self.size_to_args[self.itemsize]
        code.raw(f"auto dst_ = reinterpret_cast<{dtype_c}*>(dst);")
        code.raw("""
        asm volatile(
            "{\\n"
        """)
        pred_p = ""
        if pred:
            code.raw(f"""
            "   .reg .pred p; \\n"
            "   setp.ne.b32 p, %{args_c + 1}, 0;\\n"
            """)
            pred_p = "@p"
        if self.itemsize == 2:
            code.raw(f"""
            "   {pred_p} ld.shared.u16 %0, [%1];\\n"
            "}}\\n"
            : "=h"(*dst_)
            : "r" (ptr)
            """)
        elif self.itemsize == 4:
            code.raw(f"""
            "   {pred_p} ld.shared.u32 %0, [%1];\\n"
            "}}\\n"
            : "=r"(*dst_)
            : "r"(ptr)
            """)
        elif self.itemsize == 8:
            code.raw(f"""
            "   {pred_p} ld.shared.v2.u32 {{%0, %1}}, [%2];\\n"
            "}}\\n"
            : "=r"(dst_->x), "=r"(dst_->y)
            : "r"(ptr)
            """)
        elif self.itemsize == 16:
            code.raw(f"""
            "   {pred_p} ld.shared.v4.u32 {{%0, %1, %2, %3}}, [%4];\\n"
            "}}\\n"
            : "=r"(dst_->x), "=r"(dst_->y), "=r"(dst_->z), "=r"(dst->w)
            : "r"(ptr)
            """)
        if pred:
            code.raw(', "r"((int)pred)')
        code.raw(");")
        pass
    
    def shared_store(self, code, pred=False):
        dtype_c = self.size_to_dtype[self.itemsize]
        args_c = self.size_to_args[self.itemsize]
        code.raw(f"auto src_ = reinterpret_cast<const {dtype_c}*>(src);")
        code.raw("""
        asm volatile(
            "{\\n"
        """)
        pred_p = ""
        if pred:
            code.raw(f"""
            "   .reg .pred p; \\n"
            "   setp.ne.b32 p, %{args_c + 1}, 0;\\n"
            """)
            pred_p = "@p"
        if self.itemsize == 2:
            code.raw(f"""
            "   {pred_p} st.shared.u16 [%0], %1;\\n"
            "}}\\n"
            : : "r" (ptr), "h"(*src_)
            """)
        elif self.itemsize == 4:
            code.raw(f"""
            "   {pred_p} st.shared.u32 [%0], %1;\\n"
            "}}\\n"
            : : "r"(ptr), "r"(*src_)
            """)
        elif self.itemsize == 8:
            code.raw(f"""
            "   {pred_p} st.shared.v2.u32 [%0], {{%1, %2}};\\n"
            "}}\\n"
            : : "r"(ptr), "r"(src_->x), "r"(src_->y)
            """)
        elif self.itemsize == 16:
            code.raw(f"""
            "   {pred_p} st.shared.v4.u32 [%0], {{%1, %2, %3, %4}};\\n"
            "}}\\n"
            : : "r"(ptr), "r"(src_->x), "r"(src_->y), "r"(src_->z), "r"(src_->w)
            """)
        if pred:
            code.raw(', "r"((int)pred)')
        code.raw(");")
    
    @pccm.cuda.static_function(device=True, forceinline=True)
    def store_pred(self):
        code = pccm.code()
        code.targ('T')
        code.arg("target", "const T&")
        code.arg("ptr_", "void*")
        code.arg("pred", "int")
        code.raw("""
            uint32_t ptr = tv::gemm::get_smem_pointer(ptr_);
            auto src = &target;
        """)
        self.shared_store(code, True)
        return code
    
    @pccm.cuda.static_function(device=True, forceinline=True)
    def load_pred(self):
        code = pccm.code()
        code.targ('T')
        code.arg("dst", "T*")
        code.arg("ptr_", "const void*")
        code.arg("pred", "int")
        code.raw("""
            uint32_t ptr = tv::gemm::get_smem_pointer(ptr_);
        """)
        self.shared_load(code, True)
        return code
