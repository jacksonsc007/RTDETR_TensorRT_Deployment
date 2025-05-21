#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx_graphsurgeon as gs
import numpy as np
import onnx
import cupy as cp
import logging
import sys
import os

import tensorrt as trt
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
)

import triton
import triton.language as tl
import cupy

from enum import IntEnum

from polygraphy.json import to_json, from_json
import torch

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

def volume(d):
    return np.prod(d)

import argparse

logger = logging.getLogger("ink_plugins")

class Tactic(IntEnum):
    TORCH = 1
    TRITON = 2

@triton.jit
def circ_pad(X,
            all_pads_0, all_pads_2, all_pads_4, all_pads_6,
            orig_dims_0, orig_dims_1, orig_dims_2, orig_dims_3,
            Y,
            Y_shape_1, Y_shape_2, Y_shape_3,
            X_len, Y_len, BLOCK_SIZE: tl.constexpr,):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_y = i < Y_len

    i3 = i % Y_shape_3
    i2 = (i // Y_shape_3) % Y_shape_2
    i1 = (i // Y_shape_3 // Y_shape_2) % Y_shape_1
    i0 = i // Y_shape_3 // Y_shape_2 // Y_shape_1

    j0 = (i0 - all_pads_0 + orig_dims_0) % orig_dims_0
    j1 = (i1 - all_pads_2 + orig_dims_1) % orig_dims_1
    j2 = (i2 - all_pads_4 + orig_dims_2) % orig_dims_2
    j3 = (i3 - all_pads_6 + orig_dims_3) % orig_dims_3

    load_idx = orig_dims_3 * orig_dims_2 * orig_dims_1 * j0 + orig_dims_3 * orig_dims_2 * j1 + orig_dims_3 * j2 + j3
    mask_x = load_idx < X_len

    x = tl.load(X + load_idx, mask=mask_x)

    tl.store(Y + i, x, mask=mask_y)

PLUGIN_NAME = "fused_attn_offset_prediction"
PLUGIN_NAMESPACE = "ink_plugins"
class fused_attn_offset_prediction(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    def __init__(self, fieldCollection=None, phase=None):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.per_format_tactics = (
            False  # whether per-format tactics or global tactics should be used
        )
        self.curr_type = None  # format being timed currently by TRT auto-tuner

        self.num_outputs = 2
        self.plugin_namespace = PLUGIN_NAMESPACE
        self.plugin_name = PLUGIN_NAME
        self.plugin_version = "1"

        # Set the timing cache ID to prevent unnecessary timing of second plugin instance
        self.timing_cache_id = ""
        
        self.tactic = None

        # Deserialize weights if provided
        if fieldCollection:
            expected_field_name = ['input_scale', 'input_offset', 'sp_weight_scale', 'sp_weight_offset', 'attn_weight_scale', 'attn_weight_offset', 'int8_sp_weight', 'int8_attn_weight', 'sp_fp32_bias', 'attn_fp32_bias']
            received_field_names = [field.name for field in fieldCollection]
            assert expected_field_name == received_field_names
            self.input_scale = fieldCollection[0].data
            self.input_offset = fieldCollection[1].data
            self.sp_weight_scale = fieldCollection[2].data
            self.sp_weight_offset = fieldCollection[3].data
            self.attn_weight_scale = fieldCollection[4].data
            self.attn_weight_offset = fieldCollection[5].data
            self.int8_sp_weight = fieldCollection[6].data
            self.int8_attn_weight = fieldCollection[7].data
            self.sp_fp32_bias = fieldCollection[8].data
            self.attn_fp32_bias = fieldCollection[9].data

        if phase is not None:
            self.phase = phase

    def get_capability_interface(self, type):
        return self

    def get_output_data_types(self, input_types):
        return [trt.float32, trt.float32]

    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):

        # Input shape: [bs, n_queries, 256]
        input_dims =  trt.DimsExprs(inputs[0])
        bs = input_dims[0]
        n_queries = input_dims[1]

        attn_out = trt.DimsExprs([bs, n_queries, exprBuilder.constant(96)])
        sp_out = trt.DimsExprs([bs, n_queries, exprBuilder.constant(192)])
        return [attn_out, sp_out]

    def get_fields_to_serialize(self):
        return trt.PluginFieldCollection([
            # trt.PluginField("pads", self.pads, trt.PluginFieldType.INT32),
            trt.PluginField("input_scale", self.input_scale, trt.PluginFieldType.FLOAT32),
            trt.PluginField("input_offset", self.input_offset, trt.PluginFieldType.INT8),
            trt.PluginField("sp_weight_scale", self.sp_weight_scale, trt.PluginFieldType.FLOAT32),
            trt.PluginField("sp_weight_offset", self.sp_weight_offset, trt.PluginFieldType.INT8),
            trt.PluginField("attn_weight_scale", self.attn_weight_scale, trt.PluginFieldType.FLOAT32),
            trt.PluginField("attn_weight_offset", self.attn_weight_offset, trt.PluginFieldType.INT8),
            trt.PluginField("int8_sp_weight", self.int8_sp_weight, trt.PluginFieldType.INT8),
            trt.PluginField("int8_attn_weight", self.int8_attn_weight, trt.PluginFieldType.INT8),
            trt.PluginField("sp_fp32_bias", self.sp_fp32_bias, trt.PluginFieldType.FLOAT32),
            trt.PluginField("attn_fp32_bias", self.attn_fp32_bias, trt.PluginFieldType.FLOAT32),
            # trt.PluginField(
            #     "per_format_tactics",
            #     np.array([self.per_format_tactics], dtype=np.int32),
            #     trt.PluginFieldType.INT32,
            # ),
        ])

    def configure_plugin(self, inp, out):
        assert inp[0].desc.type == trt.int8
        self.curr_type = inp[0].desc.type

    def on_shape_change(self, inp, out):
        # if (
        #     self.phase == trt.TensorRTPhase.RUNTIME
        #     and self.per_format_tactics
        #     and inp[0].type == trt.float16
        # ):
        #     assert self.tactic == Tactic.TRITON

        # X_dims = inp[0].dims
        # self.X_shape = np.zeros((len(X_dims),))
        # for i in range(len(X_dims)):
        #     self.X_shape[i] = X_dims[i]
        pass

    def supports_format_combination(self, pos, in_out, num_inputs):
        assert num_inputs == 1
        assert pos < len(in_out)

        assert pos < len(in_out)
        d = in_out[pos].desc
        if d.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 0:
            return d.type == trt.int8
        else:
            return d.type == trt.float32

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        if self.tactic == Tactic.TORCH:
            num_outputs = len(outputs)
            assert num_outputs == self.num_outputs, "Number of outputs does not match expected number of outputs."
            input_dtype = trt.nptype(input_desc[0].type)
            output_attn_dtype = trt.nptype(output_desc[0].type)
            output_offset_dtype = trt.nptype(output_desc[1].type)
            input_mem = cupy.cuda.UnownedMemory(
                inputs[0], volume(input_desc[0].dims) * cupy.dtype(input_dtype).itemsize, self
            )
            output_attn_mem = cupy.cuda.UnownedMemory(
                outputs[0], volume(output_desc[0].dims) * cupy.dtype(output_attn_dtype).itemsize, self
            )
            output_offset_mem = cupy.cuda.UnownedMemory(
                outputs[1], volume(output_desc[1].dims) * cupy.dtype(output_offset_dtype).itemsize, self
            )
            input_ptr = cupy.cuda.MemoryPointer(input_mem, 0)
            output_attn_ptr = cupy.cuda.MemoryPointer(output_attn_mem, 0)
            output_offset_ptr = cupy.cuda.MemoryPointer(output_offset_mem, 0)
            
            input_d = cupy.ndarray(
                tuple(input_desc[0].dims),
                dtype=input_dtype,
                memptr=input_ptr,
            )
            output_attn_d = cupy.ndarray(
                tuple(output_desc[0].dims),
                dtype=output_attn_dtype,
                memptr=output_attn_ptr,
            )
            output_offset_d = cupy.ndarray(
                tuple(output_desc[1].dims),
                dtype=output_offset_dtype,
                memptr=output_offset_ptr,
            )
            
            inp_t = torch.as_tensor(input_d)
            
            # get attributes as tensors
            # Need to reshape since attributes are flattened
            int8_sp_weight_t = torch.as_tensor(self.int8_sp_weight, device="cuda").reshape(192, 256) # [192, 256]
            int8_attn_weight_t = torch.as_tensor(self.int8_attn_weight, device="cuda").reshape(96, 256) # [96, 256]
            sp_fp32_bias_t = torch.as_tensor(self.sp_fp32_bias, device="cuda") # [192]
            attn_fp32_bias_t = torch.as_tensor(self.attn_fp32_bias, device="cuda") # [96]
            input_scale_t = torch.as_tensor(self.input_scale, device="cuda") # [1]
            input_offset_t = torch.as_tensor(self.input_offset, device="cuda") # [1]
            sp_weight_scale_t = torch.as_tensor(self.sp_weight_scale, device="cuda").reshape(192, 1) # [192]
            sp_weight_offset_t = torch.as_tensor(self.sp_weight_offset, device="cuda").reshape(192, 1) # [192]
            attn_weight_scale_t = torch.as_tensor(self.attn_weight_scale, device="cuda").reshape(96, 1) # [96]
            attn_weight_offset_t = torch.as_tensor(self.attn_weight_offset, device="cuda").reshape(96, 1) # [96]
            
            # print(f"\033[96m[DEBUG] int8_sp_weight: {int8_sp_weight.shape}\033[0m")
            # print(f"\033[96m[DEBUG] int8_attn_weight: {int8_attn_weight.shape}\033[0m")
            # compute output of sampling offsets
            # print("\033[96m[INFO] Plugining running \033[0m")
            # debugging all tensors shapes
            # print(f"\033[96m[DEBUG] {output_t_sp.shape = }\033[0m")
            # print(f"\033[96m[DEBUG] {output_t_attn.shape = }\033[0m")
            # print(f"\033[96m[DEBUG] inp_t.shape: {inp_t.shape}\033[0m")
            # print(f"\033[96m[DEBUG] input_offset_t.shape: {input_offset_t.shape}\033[0m")
            # print(f"\033[96m[DEBUG] sp_weight_offset_t.shape: {sp_weight_offset_t.shape}\033[0m")
            # print(f"\033[96m[DEBUG] sp_weight_scale_t.shape: {sp_weight_scale_t.shape}\033[0m")
            # print(f"\033[96m[DEBUG] sp_fp32_bias_t.shape: {sp_fp32_bias_t.shape}\033[0m")
            # print(f"\033[96m[DEBUG] attn_weight_offset_t.shape: {attn_weight_offset_t.shape}\033[0m")
            # print(f"\033[96m[DEBUG] attn_weight_scale_t.shape: {attn_weight_scale_t.shape}\033[0m")
            # print(f"\033[96m[DEBUG] attn_fp32_bias_t.shape: {attn_fp32_bias_t.shape}\033[0m")
            # print(f"\033[96m[DEBUG] int8_sp_weight_t.shape: {int8_sp_weight_t.shape}\033[0m")
            # print(f"\033[96m[DEBUG] int8_attn_weight_t.shape: {int8_attn_weight_t.shape}\033[0m")
            
            use_int8_mul = False
            if use_int8_mul:
                # RuntimeError: Pytorch does not support `@` operator for int8 type tensor, throwing "addmm_cuda" not implemented for 'Char'
                tmp_int8_sp = (inp_t - input_offset_t) @ (int8_sp_weight_t - sp_weight_offset_t).transpose(0, 1) # [1, 300, 192]
                tmp_fp32_sp = (tmp_int8_sp) * (sp_weight_scale_t * input_scale_t)
                tmp_fp32_sp = tmp_fp32_sp + sp_fp32_bias_t

                # compute output of attention weights
                tmp_int8_attn = (inp_t - input_offset_t) @ (int8_attn_weight_t - attn_weight_offset_t).transpose(0, 1) # [1, 300, 96]
                tmp_fp32_attn = (tmp_int8_attn) * (attn_weight_scale_t * input_scale_t)
                tmp_fp32_attn = tmp_fp32_attn + attn_fp32_bias_t
            else:
                # convert back to fp32
                sp_fp32_weight = (int8_sp_weight_t - sp_weight_offset_t) * sp_weight_scale_t # (192, 256)
                attn_fp32_weight = (int8_attn_weight_t - attn_weight_offset_t) * attn_weight_scale_t # (96, 256)
                # print(sp_fp32_weight)
                # print(attn_fp32_weight)
                # print(attn_fp32_bias_t)
                input_fp32 = (inp_t - input_offset_t) * input_scale_t
                tmp_fp32_sp = input_fp32 @ sp_fp32_weight.transpose(0, 1) + sp_fp32_bias_t
                tmp_fp32_attn = input_fp32 @ attn_fp32_weight.transpose(0, 1) + attn_fp32_bias_t
            # print("\033[96m[attn]\n\033[0m", tmp_fp32_attn)
            # print("\033[96m[sp]\n\033[0m]]", tmp_fp32_sp)
            # save to local file
            # torch.save(inp_t, "plugin_input_python.pt")
            # torch.save(tmp_fp32_attn, "tmp_fp32_attn.pt")
            # torch.save(tmp_fp32_sp, "tmp_fp32_sp.pt")
                        
            
            cupy.copyto(
                output_attn_d, 
                    cupy.asarray(tmp_fp32_attn),
            )
            cupy.copyto(
                output_offset_d, 
                    cupy.asarray(tmp_fp32_sp),
            )
        else:
            raise NotImplementedError
        return 0
    
    def attach_to_context(self, context):
        return self.clone()
    
    def get_valid_tactics(self):
        assert self.curr_type is not None
        if self.per_format_tactics and self.curr_type == trt.int8:
            return [int(Tactic.TORCH)]

        return [int(Tactic.TORCH)]

    def set_tactic(self, tactic):
        self.tactic = Tactic(tactic)

        if self.phase == trt.TensorRTPhase.RUNTIME:
            logger.info(f"Best tactic chosen: {self.tactic}")

    def clone(self):
        cloned_plugin = fused_attn_offset_prediction()
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    # 
    # The following defaults take effect since the respective methods are not overriden
    #

    # def get_workspace_size(self, input_desc, output_desc):
    #     return 0
    
    # def destroy(self):
    #     pass


class fused_attn_offset_prediction_creator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = PLUGIN_NAME
        self.plugin_namespace = PLUGIN_NAMESPACE
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [
                trt.PluginField("input_scale", np.array([]), trt.PluginFieldType.FLOAT32),
                trt.PluginField("input_offset", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("sp_weight_scale", np.array([]), trt.PluginFieldType.FLOAT32),
                trt.PluginField("sp_weight_offset", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("attn_weight_scale", np.array([]), trt.PluginFieldType.FLOAT32),
                trt.PluginField("attn_weight_offset", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("int8_sp_weight", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("int8_attn_weight", np.array([]), trt.PluginFieldType.INT8),
                trt.PluginField("sp_fp32_bias", np.array([]), trt.PluginFieldType.FLOAT32),
                trt.PluginField("attn_fp32_bias", np.array([]), trt.PluginFieldType.FLOAT32),
            ]
        )

    def create_plugin(self, name, fc, phase):
        return fused_attn_offset_prediction(fc, phase)

# NOTE: manually add plugin
plg_registry = trt.get_plugin_registry()
namespace = "ink_plugins"
# namespace = ""
plg_registry.register_creator(fused_attn_offset_prediction_creator(), namespace)