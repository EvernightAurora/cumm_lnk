# Copyright 2022

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

import enum
from operator import is_
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pccm

from cumm import dtypes
from cumm.common import (GemmBasic, GemmBasicKernel, TensorView,
                         TensorViewKernel)
from cumm.gemm import (constants, gemmmath, layout, mask_iters, out_iters,
                       output_op, thread_map, volta_iters, volta_out_iters,
                       wmma)
from cumm.gemm.algospec import bases
from cumm.gemm.algospec.simt import simt_transpose_padding
from cumm.gemm.bases import (GemmApply, GemmInputIterator, GemmIterator,
                             GemmOutFragIterator, GemmOutputIterator,
                             GemmOutputOp, GemmOutSmemLoader,
                             GemmOutWarpIterator)
from cumm.gemm.core import MetaArray, metaseq, seq
from cumm.gemm.wmma.simt import WarpMmaSimt
from cumm.gemm.wmma.simt_depthwise import WarpMmaSimtDepthwise
from cumm.gemm.depthwise_wgrad_transformer import DepthwiseWgradTramsformer

from .core import GemmAlgo, ShuffleStrideType, TensorOp
from .simt import InputSimt, OutputSimt



class MmaSimtDepthwise(bases.Mma):
    def __init__(self,
                 input_spec: bases.Input,
                 tile_shape: MetaArray[int],
                 warp_tile_shape: MetaArray[int],
                 num_stage: int,
                 dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType,
                 dtype_acc: dtypes.DType,
                 trans_a: bool,
                 trans_b: bool,
                 tensorop: Optional[TensorOp] = None,
                 algo: GemmAlgo = GemmAlgo.Simt):
        self._input_spec = input_spec
        # input_sub_tile_shape_a = input_spec.
        self.input_sub_tile_shape_a = input_spec.input_sub_tile_shape_a
        self.input_sub_tile_shape_b = input_spec.input_sub_tile_shape_b

        if not trans_a:
            padding_m = simt_transpose_padding(constants.WARP_SIZE,
                                               tile_shape[2],
                                               dtype_a.bitsize())
        else:
            padding_m = 0

        if trans_b:
            padding_n = simt_transpose_padding(constants.WARP_SIZE,
                                               tile_shape[2],
                                               dtype_b.bitsize())
        else:
            padding_n = 0
        self._padding_mn = seq(padding_m, padding_n)
        self._accumulator_size = warp_tile_shape[0] * warp_tile_shape[
            1] // constants.WARP_SIZE
        warp_shape = seq(8, 4)
        if warp_tile_shape[0] <= warp_tile_shape[1]:
            warp_shape = seq(4, 8)
        if not trans_a:
            warp_shape = seq(0, min(warp_tile_shape[2], constants.WARP_SIZE))
            warp_shape[0] = constants.WARP_SIZE // warp_shape[1]
        self.warp_shape = warp_shape
        thread_mma_shape = seq(warp_tile_shape[0] // warp_shape[0],
                               warp_tile_shape[1] // warp_shape[1])
        lane_vec_load_shape = seq(constants.OPTIM_ACCESS // dtype_a.itemsize(),
                                  constants.OPTIM_ACCESS // dtype_b.itemsize())
        self.thread_mma_shape = thread_mma_shape
        # elif dtype_a == dtypes.float16 and dtype_b == dtypes.float16:
        #     lane_mma_shape = seq(
        #         min(lane_vec_load_shape[0], 4),
        #         min(lane_vec_load_shape[1], 4), 2)
        lane_mma_shape = seq(
            min(lane_vec_load_shape[0], thread_mma_shape[0]),
            min(lane_vec_load_shape[1], thread_mma_shape[1]), 1)
        self.lane_mma_shape = lane_mma_shape
        lane_interleave = 1
        self.lane_layout = layout.RowMajorInterleaved(lane_interleave)
        self.warp_gemm_iters = warp_tile_shape[2] // lane_mma_shape[
            2]  # type: int
        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()
        self.num_threads = self.warp_count * constants.WARP_SIZE
        self._partk = self.warp_count_shape[2]
        if not trans_a and self.input_sub_tile_shape_a[1] > 1:
            self._smem_iter_a = mask_iters.SmemTileIteratorVectorSplitTransposed(
               dtype_a,
                input_spec.thread_map_a,
                seq(tile_shape[2], tile_shape[0]),
                self.input_sub_tile_shape_a,
                0,
                self.num_threads,
                smem_shape=seq(0, padding_m) +
                seq(tile_shape[2] * num_stage, tile_shape[0]),
                transposed_input=not trans_a 
            )
        else:
            self._smem_iter_a = mask_iters.SmemTileIteratorV2(
                dtype_a,
                input_spec.thread_map_a,
                seq(tile_shape[2], tile_shape[0]),
                self.input_sub_tile_shape_a,
                0,
                self.num_threads,
                smem_shape=seq(0, padding_m) +
                seq(tile_shape[2] * num_stage, tile_shape[0]),
                transposed_input=not trans_a)

        self._smem_iter_b = mask_iters.SmemTileIteratorV2(
            dtype_b,
            input_spec.thread_map_b,
            seq(tile_shape[2], tile_shape[1]),
            self.input_sub_tile_shape_b,
            0,
            self.num_threads,
            smem_shape=seq(0, padding_n) +
            seq(tile_shape[2] * num_stage, tile_shape[1]),
            transposed_input=trans_b)
        self._warp_iter_a = mask_iters.WarpTileIterator(
            dtype_a,
            seq(tile_shape[2], tile_shape[0] + padding_m),
            seq(warp_tile_shape[2], warp_tile_shape[0]),
            warp_shape,
            seq(lane_mma_shape[2], lane_mma_shape[0]),
            layout.RowMajorInterleaved(self.input_sub_tile_shape_a[0]),
            self.lane_layout,
            padding_m,
            True,
            partk=self.partk)

        warp_b_is_depthwise = True
        is_depthwise_k = False
        if input_spec.trans_b:          # tnt,  Fwd
            is_depthwise_k = True
        elif not input_spec.trans_a:        # ttt,  BwdInput
            is_depthwise_k = False
        else:                           # ntt,  BwdWeight 
            warp_b_is_depthwise = False

        self._warp_iter_b = mask_iters.WarpTileIterator(
            dtype_b,
            seq(tile_shape[2], tile_shape[1] + padding_n),
            seq(warp_tile_shape[2], warp_tile_shape[1]),
            warp_shape,
            seq(lane_mma_shape[2], lane_mma_shape[1]),
            layout.RowMajorInterleaved(self.input_sub_tile_shape_b[0]),
            self.lane_layout,
            padding_n,
            False,
            partk=self.partk,
            is_depthwise_expand=warp_b_is_depthwise,
            is_depthwise_expand_k=is_depthwise_k)
        # if dtype_a == dtypes.float16 and dtype_b == dtypes.float16:
        #     self._warp_mma = WarpMmaSimt(
        #         (thread_mma_shape[0], thread_mma_shape[1], lane_mma_shape[2]),
        #         dtype_a, dtype_b, dtype_acc, False, True, False)
        # else:
        self._warp_mma = WarpMmaSimt(
            (thread_mma_shape[0], thread_mma_shape[1], lane_mma_shape[2]),
            dtype_a, dtype_b, dtype_acc, True, False, False)

    @property
    def input_spec(self) -> bases.Input:
        return self._input_spec

    @property
    def padding_mn(self):
        return self._padding_mn

    @property
    def partk(self):
        return self._partk

    @property
    def smem_iter_a(self):
        return self._smem_iter_a

    @property
    def smem_iter_b(self):
        return self._smem_iter_b

    @property
    def warp_iter_a(self):
        return self._warp_iter_a

    @property
    def warp_iter_b(self):
        return self._warp_iter_b

    @property
    def warp_mma(self):
        return self._warp_mma

    @property
    def num_warp_mma_iters(self):
        return self.warp_gemm_iters

    @property
    def accumulator_size(self) -> int:
        return self._accumulator_size


class MmaSimtDepthwiseV2(bases.Mma):
    def __init__(self,
                 input_spec: bases.Input,
                 tile_shape: MetaArray[int],
                 warp_tile_shape: MetaArray[int],
                 num_stage: int,
                 dtype_a: dtypes.DType,
                 dtype_b: dtypes.DType,
                 dtype_acc: dtypes.DType,
                 trans_a: bool,
                 trans_b: bool,
                 tensorop: Optional[TensorOp] = None,
                 algo: GemmAlgo = GemmAlgo.Simt):
        self._input_spec = input_spec
        # input_sub_tile_shape_a = input_spec.
        self.input_sub_tile_shape_a = input_spec.input_sub_tile_shape_a
        self.input_sub_tile_shape_b = input_spec.input_sub_tile_shape_b
        self.warp_count_shape = tile_shape // warp_tile_shape
        
        self._padding_mn = seq(0, 0)

        self._accumulator_size = warp_tile_shape[0] * warp_tile_shape[
            1] // constants.WARP_SIZE
        warp_shape = input_spec.warp_shape
        self.warp_shape = warp_shape
        thread_mma_shape = input_spec.thread_mma_shape
        self.thread_mma_shape = thread_mma_shape

        lane_mma_shape = input_spec.lane_mma_shape
        self.lane_mma_shape = lane_mma_shape
        self._smem_iter_a = self._smem_iter_b = None
        self._warp_iter_a = self._warp_iter_b = None
        self.lane_layout = layout.RowMajorInterleaved(1)
        self._warp_mma = WarpMmaSimtDepthwise(
            (thread_mma_shape[0], thread_mma_shape[1], lane_mma_shape[2]),
            dtype_a, dtype_b, dtype_acc, True, False, False)

    @property
    def input_spec(self) -> bases.Input:
        return self._input_spec

    @property
    def padding_mn(self):
        return self._padding_mn

    @property
    def partk(self):
        return None

    @property
    def smem_iter_a(self):
        return self._smem_iter_a

    @property
    def smem_iter_b(self):
        return self._smem_iter_b

    @property
    def warp_iter_a(self):
        return self._warp_iter_a

    @property
    def warp_iter_b(self):
        return self._warp_iter_b

    @property
    def warp_mma(self):
        return self._warp_mma

    @property
    def num_warp_mma_iters(self):
        return None

    @property
    def accumulator_size(self) -> int:
        return self._accumulator_size


class AlgoSpecificSimtDepthwise(object):
    def __init__(
            self,
            tile_shape: MetaArray[int],
            warp_tile_shape: MetaArray[int],
            num_stage: int,
            dtype_a: dtypes.DType,
            dtype_b: dtypes.DType,
            dtype_c: dtypes.DType,
            dtype_acc: dtypes.DType,
            dtype_comp: dtypes.DType,
            trans_a: bool,
            trans_b: bool,
            trans_c: bool,
            tensorop: Optional[TensorOp] = None,
            algo: GemmAlgo = GemmAlgo.Simt,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle):
        assert algo == GemmAlgo.Simt or algo == GemmAlgo.SimtDP4A
        self.input_spec = InputSimt(tile_shape, warp_tile_shape, dtype_a,
                                    dtype_b, trans_a, trans_b, algo,
                                    shuffle_stride)
        self.mma_spec = MmaSimtDepthwise(self.input_spec, tile_shape, warp_tile_shape,
                                num_stage, dtype_a, dtype_b, dtype_acc,
                                trans_a, trans_b, tensorop, algo)
        self.output_spec = OutputSimt(self.mma_spec, tile_shape,
                                      warp_tile_shape, num_stage, dtype_c,
                                      dtype_acc, dtype_comp, trans_c, tensorop,
                                      algo, shuffle_stride)