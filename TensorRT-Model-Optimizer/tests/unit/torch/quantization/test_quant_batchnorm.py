# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests of QuantBatchNorm module."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelopt.torch.quantization import set_quantizer_attribute, tensor_quant
from modelopt.torch.quantization.nn import QuantModuleRegistry

NUM_CHANNELS = 3


class TestQuantBatchNormND:
    @pytest.mark.parametrize(
        "original_cls, input_shape",
        [
            (nn.BatchNorm1d, (2, NUM_CHANNELS, 8)),
            (nn.BatchNorm2d, (2, NUM_CHANNELS, 8, 8)),
            (nn.BatchNorm3d, (2, NUM_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_no_quant(self, original_cls, input_shape):
        batchnorm_object = original_cls(NUM_CHANNELS, affine=True)
        quant_batchnorm_object = QuantModuleRegistry.convert(batchnorm_object)
        quant_batchnorm_object.input_quantizer.disable()

        test_input = torch.randn(input_shape)

        out1 = quant_batchnorm_object(test_input)
        out2 = F.batch_norm(
            test_input,
            quant_batchnorm_object.running_mean,
            quant_batchnorm_object.running_var,
            quant_batchnorm_object.weight,
            quant_batchnorm_object.bias,
            training=True,
        )
        assert torch.allclose(out1, out2, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "original_cls, input_shape",
        [
            (nn.BatchNorm1d, (2, NUM_CHANNELS, 8)),
            (nn.BatchNorm2d, (2, NUM_CHANNELS, 8, 8)),
            (nn.BatchNorm3d, (2, NUM_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_tensor(self, original_cls, input_shape):
        batchnorm_object = original_cls(NUM_CHANNELS, affine=True)
        quant_batchnorm_object = QuantModuleRegistry.convert(batchnorm_object)

        test_input = torch.randn(input_shape)
        quant_input = tensor_quant.fake_tensor_quant(test_input, torch.max(torch.abs(test_input)))

        out1 = quant_batchnorm_object(test_input)
        out2 = F.batch_norm(
            quant_input,
            quant_batchnorm_object.running_mean,
            quant_batchnorm_object.running_var,
            quant_batchnorm_object.weight,
            quant_batchnorm_object.bias,
            training=True,
        )
        assert torch.allclose(out1, out2, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "original_cls, input_shape",
        [
            (nn.BatchNorm1d, (2, NUM_CHANNELS, 8)),
            (nn.BatchNorm2d, (2, NUM_CHANNELS, 8, 8)),
            (nn.BatchNorm3d, (2, NUM_CHANNELS, 8, 8, 8)),
        ],
    )
    def test_fake_quant_per_channel(self, original_cls, input_shape):
        batchnorm_object = original_cls(NUM_CHANNELS, affine=True)
        quant_batchnorm_object = QuantModuleRegistry.convert(batchnorm_object)
        set_quantizer_attribute(quant_batchnorm_object, lambda name: True, {"axis": (1)})

        test_input = torch.randn(input_shape)
        reduce_dims = list(range(len(test_input.shape)))
        reduce_dims.pop(1)
        quant_input = tensor_quant.fake_tensor_quant(
            test_input, torch.abs(test_input).amax(dim=reduce_dims, keepdim=True)
        )

        out1 = quant_batchnorm_object(test_input)
        out2 = F.batch_norm(
            quant_input,
            quant_batchnorm_object.running_mean,
            quant_batchnorm_object.running_var,
            quant_batchnorm_object.weight,
            quant_batchnorm_object.bias,
            training=True,
        )
        assert torch.allclose(out1, out2, rtol=0, atol=0)
