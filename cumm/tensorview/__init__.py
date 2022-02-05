# Copyright 2022 Yan Yan
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
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pccm import Argument
from pccm.middlewares.pybind import (TemplateTypeStmt,
                                     _simple_template_type_parser)
from torch import tensor

from cumm.core_cc import tensorview_bind
from cumm.core_cc.tensorview_bind import CUDAKernelTimer
from cumm.core_cc.tensorview_bind import NVRTCModule as _NVRTCModule
from cumm.core_cc.tensorview_bind import NVRTCProgram, Tensor
from . import gemm 

from cumm.dtypes import get_npdtype_from_tvdtype

bool_ = 0
float16 = 1
float32 = 2
float64 = 3
int8 = 4
int16 = 5
int32 = 6
int64 = 7
uint8 = 8
uint16 = 9
uint32 = 10
uint64 = 11
tf32 = 13

custom16 = 100
custom32 = 101
custom48 = 102
custom64 = 103
custom80 = 104
custom96 = 105
custom128 = 106

_SIMPLE_TYPES_TO_TV_DTYPE = {
    "int": int32,
    "int8_t": int8,
    "int16_t": int16,
    "int32_t": int32,
    "int64_t": int64,
    "uint8_t": uint8,
    "uint16_t": uint16,
    "uint32_t": uint32,
    "uint64_t": uint64,
    "std::intptr_t": int64,
    "std::uintptr_t": uint64,
    "size_t": uint64,
    "std::size_t": uint64,
    "unsigned": uint32,
    "long": int64,
    "short": int16,
    "float": float32,
    "double": float64,
    "unsigned long": uint64,
    "unsigned int": uint32,
    "__half": float16,
    "tv::half_t": float16,
    "at::Half": float16,
}  # type: Dict[str, int]

_VALID_CONTAINER_TYPES = {"tv::array", "std::array"}


@dataclass
class NVRTCArgMeta:
    valid: bool
    simple_type: int
    shape: List[int]

    is_simple_ptr: bool = False


class NVRTCKernelMeta:
    def __init__(self, name: str, ns: str, args: List[Argument]):
        self.name = name
        self.ns = ns
        self.args = args
        self.arg_types = [
            _simple_template_type_parser(a.type_str, {}) for a in args
        ]
        self.simple_types: List[Optional[int]] = []
        self.arg_metas: List[NVRTCArgMeta] = []
        for meta in self.arg_types:
            simple_tv_type = -1
            is_simple_ptr = False
            valid = meta.name != ""
            shape: List[int] = []
            if valid and not meta.is_ptr:
                # determine scalar type
                cur_meta = meta
                while True:
                    if len(cur_meta.args) == 0:
                        if cur_meta.name in _SIMPLE_TYPES_TO_TV_DTYPE:
                            simple_tv_type = _SIMPLE_TYPES_TO_TV_DTYPE[
                                cur_meta.name]
                        break
                    is_valid_container = cur_meta.name in _VALID_CONTAINER_TYPES
                    if not is_valid_container and len(cur_meta.args) > 0:
                        valid = False
                        break
                    length = int(cur_meta.args[1].name)
                    shape.append(length)
                    cur_meta = cur_meta.args[0]
            elif valid and meta.is_ptr:
                if meta.name in _SIMPLE_TYPES_TO_TV_DTYPE:
                    is_simple_ptr = True
                    simple_tv_type = _SIMPLE_TYPES_TO_TV_DTYPE[meta.name]
            if len(shape) == 0:
                shape = [1]
            # shape = shape[::-1]
            self.arg_metas.append(
                NVRTCArgMeta(valid, simple_tv_type, shape, is_simple_ptr))

    def __repr__(self) -> str:
        return f"NVRTCKernelMeta[name={self.name},ns={self.ns},args={self.arg_metas}]"


class NVRTCModule:
    def __init__(self,
                 code: Union[str, NVRTCProgram],
                 headers: Optional[Dict[str, str]] = None,
                 opts: Optional[List[str]] = None,
                 program_name: str = "kernel.cu",
                 name_exprs: Optional[List[str]] = None,
                 name_to_meta: Optional[Dict[str, NVRTCKernelMeta]] = None,
                 cudadevrt_path: str = "") -> None:
        if headers is None:
            headers = {}
        if opts is None:
            opts = []
        if name_exprs is None:
            name_exprs = []
        if isinstance(code, str):
            self._mod = _NVRTCModule(code, headers, opts, program_name,
                                     name_exprs, cudadevrt_path)
        else:
            self._mod = _NVRTCModule(code, cudadevrt_path)
        self.blocks = [0, 0, 0]
        self.threads = [0, 0, 0]
        self.smem_size = 0
        self.stream = 0
        self._name_exprs = name_exprs
        self.name_to_meta = name_to_meta

    def load(self):
        return self._mod.load()

    def get_cpp_object(self):
        return self._mod

    def get_ptx(self):
        return self._mod.program.ptx()

    def prepare_launch(self,
                       blocks: List[int],
                       threads: List[int],
                       smem_size: int = 0,
                       stream: int = 0):
        self.blocks = blocks
        self.threads = threads
        self.smem_size = smem_size
        self.stream = stream
        return self

    def get_kernel_attrs(self, name: str):
        return self._mod.get_kernel_attributes(name)

    def run_kernel(self, name: str,
                   *args: Union[Tensor, int, float, List[int], List[float],
                                Tuple[float, ...], Tuple[int, ...]]):
        assert np.prod(self.blocks) > 0
        assert np.prod(self.threads) > 0
        metas: List[NVRTCArgMeta] = [NVRTCArgMeta(False, -1, [])] * len(args)
        if self.name_to_meta:
            assert name in self.name_to_meta, f"can't find your kernel {name}, available: {self.name_to_meta.keys()}"
            assert len(args) == len(self.name_to_meta[name].args)
            metas = self.name_to_meta[name].arg_metas
        if self._name_exprs:
            name = self.get_lowered_name(name)

        kernel_args: List[Tuple[Tensor, int]] = []
        for arg, meta in zip(args, metas):
            if meta.valid:
                # print(meta.shape)
                if meta.is_simple_ptr:
                    if not isinstance(arg, Tensor):
                        raise ValueError("your arg must be tensor")
                    if not arg.dtype == meta.simple_type:
                        cur_dtype = get_npdtype_from_tvdtype(arg.dtype)
                        expected_dtype = get_npdtype_from_tvdtype(
                            meta.simple_type)
                        raise ValueError(
                            f"your tensor {arg.shape}|{cur_dtype}"
                            f" dtype not equal to {expected_dtype}")
                    kernel_args.append((arg, _NVRTCModule.kTensor))
                    continue
                else:
                    # we can't ensure arg isn't tv::Tensor.
                    if not isinstance(arg, Tensor):
                        assert not isinstance(arg, Tensor)
                        arg_array = np.array(arg)
                        if not arg_array.shape:
                            arg_array = arg_array.reshape(1)
                        assert list(arg_array.shape) == meta.shape
                        # auto dtype cast
                        # TODO prevent floats assigned to ints
                        ten = empty(meta.shape, meta.simple_type, -1)
                        ten.numpy_view()[:] = arg_array
                        kernel_args.append((ten, _NVRTCModule.kScalar))
                        continue
            # meta isn't valid, use regular dtypes.
            if isinstance(arg, (int, float)):
                dtype = float32
                if isinstance(arg, int):
                    dtype = int64
                ten = empty([1], dtype, -1)
                ten.numpy_view()[0] = arg
                kernel_args.append((ten, _NVRTCModule.kScalar))
            elif isinstance(arg, (list, tuple)):
                dtype = np.float32
                if isinstance(arg[0], int):
                    dtype = np.int64
                arg_np = np.array(arg, dtype=dtype)
                ten = from_numpy(arg_np).clone()
                kernel_args.append((ten, _NVRTCModule.kArray))
            else:
                assert isinstance(arg, Tensor)
                kernel_args.append((arg, _NVRTCModule.kTensor))

        return self._mod.run_kernel(name, self.blocks, self.threads,
                                    self.smem_size, self.stream, kernel_args)

    @property
    def program(self):
        return self._mod.program

    def get_lowered_name(self, name: str) -> str:
        return self._mod.get_lowered_name(name)


class nullcontext(contextlib.AbstractContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


class KernelTimer:
    def __init__(self, enable: bool = True) -> None:
        self.enable = enable and not tensorview_bind.is_cpu_only()
        if self.enable:
            self._timer = CUDAKernelTimer(enable)
        else:
            self._timer = None

    def get_cpp_object(self):
        assert self._timer is not None 
        return self._timer

    @contextlib.contextmanager
    def _namespace(self, name: str):
        assert self._timer is not None
        self._timer.push(name)
        try:
            yield
        finally:
            self._timer.pop()

    @contextlib.contextmanager
    def _record(self, name: str, stream: int = 0, exit_handler: Optional[Callable[["CUDAKernelTimer", str], None]] = None):
        assert self._timer is not None
        self._timer.push(name)
        try:
            pair_name = self._timer.insert_pair("", "start", "stop")
            self._timer.record("start", stream)
            yield
            self._timer.record("stop", stream)
            if exit_handler is not None:
                exit_handler(self._timer, pair_name)
        finally:
            self._timer.pop()

    def namespace(self, name: str):
        if self.enable:
            return self._namespace(name)
        else:
            return nullcontext()

    def record(self, name: str, stream: int = 0):
        if self.enable:
            return self._record(name, stream)
        else:
            return nullcontext()

    def get_all_pair_time(self) -> Dict[str, float]:
        if self.enable:
            assert self._timer is not None
            return self._timer.get_all_pair_duration()
        else:
            return {}

    @staticmethod
    def collect_by_name(name: str, res: Dict[str, float]):
        filtered_res: Dict[str, float] = {}
        for k, v in res.items():
            k_split = k.split(".")
            if name in k_split:
                filtered_res[k] = v
        return filtered_res

def _print_exit_handler(tim: CUDAKernelTimer, name: str):
    duration = tim.get_pair_duration(name)
    print(f"{name}: {duration}")

def measure_and_print(name: str = "CUDATimer", stream: int = 0):
    tim = KernelTimer()
    return tim._record(name, stream, _print_exit_handler)

def get_numpy_view(ten: Tensor) -> np.ndarray:
    if not ten.is_contiguous():
        raise NotImplementedError(
            "numpy_view only support contiguous tv::Tensor")
    buf = ten.get_memoryview()
    return np.frombuffer(buf, dtype=TENSOR_TO_NPDTYPE_MAP[ten.dtype]).reshape(
        ten.shape)


def numpy_view(self):
    return get_numpy_view(self)


Tensor.numpy_view = numpy_view

NPDTYPE_TO_TENSOR_MAP = {
    np.dtype(np.float32): float32,
    np.dtype(np.int32): int32,
    np.dtype(np.int16): int16,
    np.dtype(np.int8): int8,
    np.dtype(np.float64): float64,
    np.dtype(np.bool_): bool_,
    np.dtype(np.uint8): uint8,
    np.dtype(np.float16): float16,
    np.dtype(np.int64): int64,
    np.dtype(np.uint16): uint16,
    np.dtype(np.uint32): uint32,
    np.dtype(np.uint64): uint64,
}

ALL_TV_TENSOR_DTYPES = set([
    bool_, float16, float32, float64, int8, int16, int32, int64, uint8, uint16,
    uint32, uint64, tf32, custom16, custom32, custom48, custom64, custom80,
    custom96, custom128
])

TENSOR_TO_NPDTYPE_MAP = {v: k for k, v in NPDTYPE_TO_TENSOR_MAP.items()}
TENSOR_TO_NPDTYPE_MAP[tf32] = np.dtype(np.float32)

_SUPPORTED_FILL_INT = {int32, int16, int8, uint32, uint16, uint8}


def zeros(shape: List[int],
          dtype: Union[np.dtype, int] = np.float32,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.zeros(shape, tv_dtype, device, pinned, managed)


def from_blob_strided(ptr: int, shape: List[int], stride: List[int],
                      dtype: Union[np.dtype, int], device: int) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_blob(ptr, shape, stride, tv_dtype, device)


def from_const_blob_strided(ptr: int, shape: List[int], stride: List[int],
                            dtype: Union[np.dtype,
                                         int], device: int) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_const_blob(ptr, shape, stride, tv_dtype,
                                           device)


def from_blob(ptr: int, shape: List[int], dtype: Union[np.dtype, int],
              device: int) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_blob(ptr, shape, tv_dtype, device)


def from_const_blob(ptr: int, shape: List[int], dtype: Union[np.dtype, int],
                    device: int) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in ALL_TV_TENSOR_DTYPES
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.from_const_blob(ptr, shape, tv_dtype, device)


def empty(shape: List[int],
          dtype: Union[np.dtype, int] = np.float32,
          device: int = -1,
          pinned: bool = False,
          managed: bool = False) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in TENSOR_TO_NPDTYPE_MAP
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.empty(shape, tv_dtype, device, pinned, managed)


def full(shape: List[int],
         val: Union[int, float],
         dtype: Union[np.dtype, int] = np.float32,
         device: int = -1,
         pinned: bool = False,
         managed: bool = False) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in TENSOR_TO_NPDTYPE_MAP
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    if tv_dtype == float32:
        return tensorview_bind.full_float(shape, val, tv_dtype, device, pinned,
                                          managed)
    elif tv_dtype in _SUPPORTED_FILL_INT:
        return tensorview_bind.full_int(shape, val, tv_dtype, device, pinned,
                                        managed)
    else:
        raise NotImplementedError


def zeros_managed(shape: List[int],
                  dtype: Union[np.dtype, int] = np.float32) -> Tensor:
    if isinstance(dtype, int):
        assert dtype in TENSOR_TO_NPDTYPE_MAP
        tv_dtype = dtype
    else:
        tv_dtype = NPDTYPE_TO_TENSOR_MAP[np.dtype(dtype)]
    return tensorview_bind.zeros_managed(shape, tv_dtype)


def from_numpy(arr: np.ndarray) -> Tensor:
    return tensorview_bind.from_numpy(arr)


def get_compute_capability(index: int):
    return tensorview_bind.get_compute_capability(index)


def is_cpu_only():
    return tensorview_bind.is_cpu_only()

