import pccm 
from cumm.gemm import constants, core
from cumm import cudasim
from typing import List, Tuple, Optional
import numpy as np 
from cumm import dtypes


class GlobalLoad(pccm.ParameterizedClass):
    def __init__(self, load_bytes: int, cache_op: Optional[pccm.cuda.CacheOpLd] = None, prefetch: bool = False):
        super().__init__()
        self.load_bytes = load_bytes
        self.cache_op = cache_op
        self.load_dtype = dtypes.uint32
        self.prefetch = prefetch
        if load_bytes >= 4:
            self.count = self.load_bytes // 4
            assert self.load_bytes % 4 == 0
            self.fragment_t = core.array_type(str(self.load_dtype), self.load_bytes // 4)
        else:
            self.count = 1
            self.load_dtype = dtypes.uint16
            if load_bytes == 2:
                self.load_dtype = dtypes.uint8
            self.fragment_t = core.array_type(str(self.load_dtype), 1)

    @pccm.cuda.static_function(device=True, forceinline=True)
    def run(self):
        code = pccm.cuda.PTXCode()
        code.targ("Frag")
        code.arg("frag", f"Frag &")
        code.arg("ptr", "void const*")
        code.arg("pred", "bool")
        code.raw(f"{self.load_dtype}* frag_ptr = reinterpret_cast<{self.load_dtype}*>(&frag);")
        with code.asm_block() as asm:
            ptr_addr = asm.global_ptr("ptr")
            frag = asm.reg_ptr("frag_ptr", pccm.cuda.RegDType.B32)
            pred = asm.ext_reg("(int)pred", pccm.cuda.RegDType.B32)
            for i in range(self.count):
                asm.mov(frag[i], frag[i]) # TODO WTF???

            with asm.pred_if("p", "ne", pred, 0) as reg:
                frag_unpack = frag.unpack(self.count)
                if self.count > 4:
                    num_vec_load = self.count // 4
                    for i in range(num_vec_load):
                        if self.prefetch:
                            asm.generic("prefetch.global.L2", [ptr_addr + i * 16])
                        asm.ld(ptr_addr + i * 16, frag_unpack[i * 4:(i + 1) * 4], self.cache_op)
                else:
                    if self.prefetch:
                        asm.generic("prefetch.global.L2", [ptr_addr])
                    asm.ld(ptr_addr, frag_unpack, self.cache_op)

        return code 