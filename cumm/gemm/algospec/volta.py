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

import enum
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
from cumm.gemm.bases import (GemmApply, GemmInputIterator, GemmIterator,
                             GemmOutFragIterator, GemmOutputIterator,
                             GemmOutputOp, GemmOutSmemLoader,
                             GemmOutWarpIterator)
from cumm.gemm.core import MetaArray, metaseq, seq
from cumm.gemm.wmma.simt import WarpMmaSimt

from .core import GemmAlgo, ShuffleStrideType, TensorOp


class InputVolta(bases.Input):
    def __init__(
            self,
            tile_shape: MetaArray[int],
            warp_tile_shape: MetaArray[int],
            dtype_a: dtypes.DType,
            dtype_b: dtypes.DType,
            trans_a: bool,
            trans_b: bool,
            algo: GemmAlgo = GemmAlgo.Volta,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            tensor_op: Optional[TensorOp] = None):
        self._trans_a = trans_a
        self._trans_b = trans_b

        m = tile_shape[0]
        n = tile_shape[1]
        k = tile_shape[2]
        self._tile_shape = tile_shape

        self.input_tile_shape_a = seq(m, k)
        if trans_a:
            self.input_tile_shape_a = seq(k, m)
        self.input_tile_shape_b = seq(k, n)
        if trans_b:
            self.input_tile_shape_b = seq(n, k)
        self.advance_axis_a = 0 if trans_a else 1
        self.advance_axis_b = 1 if trans_b else 0
        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()
        self.num_threads = self.warp_count * constants.WARP_SIZE
        self.alignment_a = constants.OPTIM_ACCESS // dtype_a.itemsize()
        self.alignment_b = constants.OPTIM_ACCESS // dtype_b.itemsize()
        self.input_sub_tile_shape_a = seq(1, self.alignment_a)
        self.input_sub_tile_shape_b = seq(1, self.alignment_b)
        warp_shape_raked_a = seq(8, 4)
        warp_shape_raked_b = seq(4, 8)
        if trans_a:
            warp_shape_raked_a = warp_shape_raked_a[::-1]
        if trans_b:
            warp_shape_raked_b = warp_shape_raked_b[::-1]

        self.tmap_a = thread_map.PitchLinearWarpRaked(
            self.input_tile_shape_a, self.input_sub_tile_shape_a,
            warp_shape_raked_a, self.num_threads)
        self.tmap_b = thread_map.PitchLinearWarpRaked(
            self.input_tile_shape_b, self.input_sub_tile_shape_b,
            warp_shape_raked_b, self.num_threads)
        self.padding_mn = seq(0, 0)
        interleaved_wmma_shape = seq(32, 32, 4)
        self.warp_gemm_iters = warp_tile_shape[2] // interleaved_wmma_shape[2]

        self.input_trans_load_a = False
        self.input_trans_load_b = False
        self.input_last_residual = True
        shuffle_a = shuffle_stride != ShuffleStrideType.NoShuffle
        shuffle_b = shuffle_stride == ShuffleStrideType.ShuffleAB

        inp_iter_a_param = mask_iters.MaskTileIteratorParams(
            dtype_a, self.input_tile_shape_a, self.input_sub_tile_shape_a,
            self.tmap_a, self.advance_axis_a, shuffle_a)
        inp_iter_b_param = mask_iters.MaskTileIteratorParams(
            dtype_b, self.input_tile_shape_b, self.input_sub_tile_shape_b,
            self.tmap_b, self.advance_axis_b, shuffle_b)

        self.inp_iter_a = mask_iters.MaskTileIterator(
            dtype_a, self.input_tile_shape_a, self.input_sub_tile_shape_a,
            self.tmap_a, inp_iter_a_param, self.advance_axis_a,
            self.alignment_a, self.input_trans_load_a,
            self.input_last_residual, shuffle_a)

        self.inp_iter_b = mask_iters.MaskTileIterator(
            dtype_b, self.input_tile_shape_b, self.input_sub_tile_shape_b,
            self.tmap_b, inp_iter_b_param, self.advance_axis_b,
            self.alignment_b, self.input_trans_load_b,
            self.input_last_residual, shuffle_b)

    @property
    def input_iter_a(self) -> GemmInputIterator:
        return self.inp_iter_a

    @property
    def input_iter_b(self) -> GemmInputIterator:
        return self.inp_iter_b

    @property
    def trans_a(self) -> bool:
        return self._trans_a

    @property
    def trans_b(self) -> bool:
        return self._trans_b

    @property
    def tile_shape(self) -> MetaArray[int]:
        return self._tile_shape


class MmaVolta(bases.Mma):
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
                 tensorop: TensorOp,
                 algo: GemmAlgo = GemmAlgo.Turing):
        self._input_spec = input_spec
        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()
        self.num_threads = self.warp_count * constants.WARP_SIZE
        tile_shape_km = seq(tile_shape[2], tile_shape[0])
        tile_shape_kn = seq(tile_shape[2], tile_shape[1])
        warp_tile_shape_km = seq(warp_tile_shape[2], warp_tile_shape[0])
        warp_tile_shape_kn = seq(warp_tile_shape[2], warp_tile_shape[1])
        self._accumulator_size = warp_tile_shape[0] * warp_tile_shape[
            1] // constants.WARP_SIZE
        interleaved_wmma_shape = seq(32, 32, 4)
        inst_shape = seq(16, 16, 4)
        self.warp_gemm_iters = warp_tile_shape[2] // interleaved_wmma_shape[2]
        self._partk = self.warp_count_shape[2]

        if trans_a:
            self._smem_iter_a = volta_iters.VoltaSmemTileIteratorCongruous(
                dtype_a, True, tile_shape_km, input_spec.input_iter_a.tmap,
                num_stage)
            self._warp_iter_a = volta_iters.VoltaWarpTileIteratorCongruous(
                dtype_a, tile_shape_km, warp_tile_shape_km, True)
        else:
            self._smem_iter_a = volta_iters.VoltaSmemTileIteratorCrosswise(
                dtype_a, tile_shape_km, input_spec.input_iter_a.tmap,
                num_stage)
            self._warp_iter_a = volta_iters.VoltaWarpTileIteratorCrosswise(
                dtype_a, tile_shape_km, warp_tile_shape_km, True)

        if trans_b:
            self._smem_iter_b = volta_iters.VoltaSmemTileIteratorCrosswise(
                dtype_b, tile_shape_kn, input_spec.input_iter_b.tmap,
                num_stage)
            self._warp_iter_b = volta_iters.VoltaWarpTileIteratorCrosswise(
                dtype_b, tile_shape_kn, warp_tile_shape_kn, False)
        else:
            self._smem_iter_b = volta_iters.VoltaSmemTileIteratorCongruous(
                dtype_b, False, tile_shape_kn, input_spec.input_iter_b.tmap,
                num_stage)
            self._warp_iter_b = volta_iters.VoltaWarpTileIteratorCongruous(
                dtype_b, tile_shape_kn, warp_tile_shape_kn, False)
        self._warp_mma = wmma.volta.WarpMmaVolta(
            (warp_tile_shape[0], warp_tile_shape[1],
             interleaved_wmma_shape[2]), dtype_a, dtype_b, dtype_acc, trans_a,
            trans_b, False)

    @property
    def input_spec(self) -> bases.Input:
        return self._input_spec

    @property
    def padding_mn(self):
        return seq(0, 0)

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


class OutputVolta(bases.Output):
    def __init__(
            self,
            mma_spec: MmaVolta,
            tile_shape: MetaArray[int],
            warp_tile_shape: MetaArray[int],
            num_stage: int,
            dtype_c: dtypes.DType,
            dtype_acc: dtypes.DType,
            dtype_comp: dtypes.DType,
            trans_c: bool,
            tensorop: Optional[TensorOp] = None,
            algo: GemmAlgo = GemmAlgo.Simt,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle,
            access_per_vector: int = 1):
        self._mma_spec = mma_spec
        output_op_count = constants.OPTIM_ACCESS_BITS // dtype_c.bitsize()
        # 32 * 16 * (128 * 4096 + 256 * 4096) * 2
        self.warp_count_shape = tile_shape // warp_tile_shape
        self.warp_count = self.warp_count_shape.prod()

        output_op_count = constants.OPTIM_ACCESS_BITS // dtype_c.bitsize()
        self.output_op_count = output_op_count

        self.acc_frag_iter = volta_out_iters.OutFragIterVolta(
            dtype_acc, warp_tile_shape)

        self.out_warp_tile_iter = volta_out_iters.OutWarpTileIteratorVolta(
            dtype_acc, tile_shape, warp_tile_shape)
        self.out_smem_padding = self.out_warp_tile_iter.padding
        self.part_shape = seq(1, self.warp_count_shape[0], 4, 4, tile_shape[1])
        # TODO handle constant 16
        self.part_dilation = seq(warp_tile_shape[0] // 16, 1,
                                 warp_tile_shape[0] // 32, 2, 1)
        self.smem_part_dilation = seq(1, 1, 1, 1, 1)
        self._num_out_iters = warp_tile_shape[0] // 16

        self.out_tmap = thread_map.Out5DLinear(dtype_acc, self.part_shape,
                                               self.part_dilation,
                                               self.warp_count,
                                               output_op_count)
        self.out_smem_tmap = thread_map.Out5DLinear(dtype_acc, self.part_shape,
                                                    self.smem_part_dilation,
                                                    self.warp_count,
                                                    output_op_count)
        # self.shared_mem_alignment = dtype_acc.bitsize(
        # ) * self.out_warp_tile_iter.element_per_acc // 8
        self.shared_mem_alignment = dtype_acc.bitsize(
        ) * self.out_warp_tile_iter.element_per_acc // 8
        assert self.shared_mem_alignment == 8
        self.out_smem_loader = out_iters.OutSmemLoader(
            dtype_acc, self.out_smem_tmap, output_op_count,
            tile_shape[1] + self.out_smem_padding[1],
            self.shared_mem_alignment)

        self._frag_per_iter = 1
        shuffle = shuffle_stride == ShuffleStrideType.ShuffleAC
        out_iter_params = out_iters.OutIteratorParams(self.out_tmap, shuffle)

        self._out_iter = out_iters.OutIterator(
            dtype_c,
            self.out_tmap,
            out_iter_params,
            self.part_shape,
            self.part_dilation,
            output_op_count,
            shuffle_in_stride=shuffle,
            access_per_vector=access_per_vector)
        self._const_out_iter = out_iters.OutIterator(
            dtype_c,
            self.out_tmap,
            out_iter_params,
            self.part_shape,
            self.part_dilation,
            output_op_count,
            read_only=True,
            shuffle_in_stride=shuffle,
            access_per_vector=access_per_vector)

        self.out_unary_op_fp_t = f"tv::math::UnaryIdentity<{dtype_comp}, {output_op_count}>"
        self.out_unary_op_i8_t = f"tv::math::Clamp<{dtype_comp}, {dtype_c}, {output_op_count}>"
        out_unary_op_fp = gemmmath.UnaryActivation(dtype_comp, dtype_c,
                                                 output_op_count)
        out_unary_op_i8 = gemmmath.Clamp(dtype_comp, dtype_c, output_op_count)

        if dtype_c == dtypes.int8:
            self.out_unary_op_t = self.out_unary_op_i8_t
            self.out_unary_op = out_unary_op_i8
        else:
            self.out_unary_op_t = self.out_unary_op_fp_t
            self.out_unary_op = out_unary_op_fp
        self._output_op = output_op.linear.LinearCombination(
            dtype_c, output_op_count, dtype_acc, dtype_comp, self.out_unary_op)
        self._apply_op = output_op.apply.ApplyOutputOp(
            output_op_count, self.output_op, self.out_iter.fragment_t,
            self.out_smem_loader.fragment_t)

    @property
    def mma_spec(self) -> bases.Mma:
        return self._mma_spec

    @property
    def smem_padding(self) -> MetaArray[int]:
        return self.out_smem_padding

    @property
    def frag_iter(self) -> GemmOutFragIterator:
        return self.acc_frag_iter

    @property
    def warp_store_iter(self) -> GemmOutWarpIterator:
        return self.out_warp_tile_iter

    @property
    def smem_loader(self) -> GemmOutSmemLoader:
        return self.out_smem_loader

    @property
    def out_iter(self) -> GemmOutputIterator:
        return self._out_iter

    @property
    def const_out_iter(self) -> GemmOutputIterator:
        return self._const_out_iter

    @property
    def num_out_iters(self) -> int:
        return self._num_out_iters

    @property
    def frag_per_iter(self) -> int:
        return self._frag_per_iter

    @property
    def output_op(self) -> GemmOutputOp:
        return self._output_op

    @property
    def apply_op(self) -> GemmApply:
        return self._apply_op


class AlgoSpecificVolta(object):
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
            algo: GemmAlgo = GemmAlgo.Volta,
            shuffle_stride: ShuffleStrideType = ShuffleStrideType.NoShuffle):
        assert algo == GemmAlgo.Volta
        self.input_spec = InputVolta(tile_shape, warp_tile_shape, dtype_a,
                                     dtype_b, trans_a, trans_b, algo,
                                     shuffle_stride)
        self.mma_spec = MmaVolta(self.input_spec, tile_shape, warp_tile_shape,
                                 num_stage, dtype_a, dtype_b, dtype_acc,
                                 trans_a, trans_b, tensorop, algo)
        self.output_spec = OutputVolta(self.mma_spec, tile_shape,
                                       warp_tile_shape, num_stage, dtype_c,
                                       dtype_acc, dtype_comp, trans_c,
                                       tensorop, algo, shuffle_stride)
