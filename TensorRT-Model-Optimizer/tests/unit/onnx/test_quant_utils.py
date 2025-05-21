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

import numpy as np
import torch
from onnx.helper import pack_float32_to_4bit

from modelopt.onnx.quantization.quant_utils import (
    pack_float32_to_4bit_cpp_based,
    pack_float32_to_4bit_optimized,
)


def _validate_results(expected_values, observed_values):
    assert len(expected_values) == len(observed_values), "length-mismatch"
    for i in range(len(expected_values)):
        assert expected_values[i] == observed_values[i], "data-mismatch"


def test_pack_float32_to_4bit_utils():
    input_pattern = [-123.4, 2.3, 0.23, 12345.1, -20123.4, 256.7, 0.83, -1.54]

    # test-case-1: Signed = True, input-length = even
    test_output10 = pack_float32_to_4bit(input_pattern, True)
    test_output11 = pack_float32_to_4bit_optimized(input_pattern, True)
    test_output12 = pack_float32_to_4bit_cpp_based(input_pattern, True)
    _validate_results(test_output10, test_output11)
    _validate_results(test_output10, test_output12)

    # test-case-2: Signed = False, input-length = even
    test_output20 = pack_float32_to_4bit(input_pattern, False)
    test_output21 = pack_float32_to_4bit_optimized(input_pattern, False)
    test_output22 = pack_float32_to_4bit_cpp_based(input_pattern, False)
    _validate_results(test_output20, test_output21)
    _validate_results(test_output20, test_output22)

    # test-case-3: Signed = True, input-length = odd
    test_output30 = pack_float32_to_4bit(input_pattern[:-1], True)
    test_output31 = pack_float32_to_4bit_optimized(input_pattern[:-1], True)
    test_output32 = pack_float32_to_4bit_cpp_based(input_pattern[:-1], True)
    _validate_results(test_output30, test_output31)
    _validate_results(test_output30, test_output32)

    # test-case-4: Signed = False, input-length = odd
    test_output40 = pack_float32_to_4bit(input_pattern[:-1], False)
    test_output41 = pack_float32_to_4bit_optimized(input_pattern[:-1], False)
    test_output42 = pack_float32_to_4bit_cpp_based(input_pattern[:-1], False)
    _validate_results(test_output40, test_output41)
    _validate_results(test_output40, test_output42)

    # test-case-5: Signed=True, input-length = 1
    test_output50 = pack_float32_to_4bit(input_pattern[0:1], True)
    test_output51 = pack_float32_to_4bit_optimized(input_pattern[0:1], True)
    test_output52 = pack_float32_to_4bit_cpp_based(input_pattern[0:1], True)
    _validate_results(test_output50, test_output51)
    _validate_results(test_output50, test_output52)

    # test-case-6: Signed=True, input = m x n float array (i.e. 2D input)
    m = 4  # m rows
    n = 8  # n columns
    input_2d = [[input_pattern[i % len(input_pattern)] for i in range(n)] for i in range(m)]
    tensor_2d = np.array(input_2d, dtype=np.float32)
    test_output60 = pack_float32_to_4bit(tensor_2d, True)
    test_output61 = pack_float32_to_4bit_optimized(tensor_2d, True)
    test_output62 = pack_float32_to_4bit_cpp_based(tensor_2d, True)
    _validate_results(test_output60, test_output61)
    _validate_results(test_output60, test_output62)

    # test-case-7: Signed=True, input = 1D numpy array of size 8
    np_array = np.array(input_pattern, dtype=np.float32)
    test_output70 = pack_float32_to_4bit(np_array, True)
    test_output71 = pack_float32_to_4bit_optimized(np_array, True)
    test_output72 = pack_float32_to_4bit_cpp_based(np_array, True)
    _validate_results(test_output70, test_output71)
    _validate_results(test_output70, test_output72)

    # test-case-8: Signed=True, input = 1D tensor of size 8
    input_tensor = torch.Tensor(input_pattern)
    test_output80 = pack_float32_to_4bit(input_tensor, True)
    test_output81 = pack_float32_to_4bit_optimized(input_tensor, True)
    test_output82 = pack_float32_to_4bit_cpp_based(input_tensor, True)
    _validate_results(test_output80, test_output81)
    _validate_results(test_output80, test_output82)

    input_pattern_int8 = [123, 2, 1, -23, -3, -127, 8, 127]
    np8 = np.asarray(input_pattern_int8, dtype=np.int8)
    np8_odd = np.asarray(input_pattern_int8[:-1], dtype=np.int8)

    # test-case-9: Signed=True, input = numpy array of dtype int8, size = even
    test_output91 = pack_float32_to_4bit_optimized(np8, True)
    test_output92 = pack_float32_to_4bit_cpp_based(np8, True)
    test_output93 = pack_float32_to_4bit(np8, True)
    _validate_results(test_output91, test_output92)
    _validate_results(test_output91, test_output93)

    # test-case-10: Signed=False, input = numpy array of dtype int8, size = odd
    test_output1001 = pack_float32_to_4bit_optimized(np8_odd, False)
    test_output1002 = pack_float32_to_4bit_cpp_based(np8_odd, False)
    test_output1003 = pack_float32_to_4bit(np8_odd, False)
    _validate_results(test_output1001, test_output1002)
    _validate_results(test_output1001, test_output1003)

    input_pattern_uint8 = [123, 2, 1, 56, 127, 13, 5, 15]
    npu8 = np.asarray(input_pattern_uint8, dtype=np.uint8)
    npu8_odd = np.asarray(input_pattern_uint8[:-1], dtype=np.uint8)

    # test-case-11: Signed=True, input = numpy array of dtype uint8, size = even
    test_output111 = pack_float32_to_4bit_optimized(npu8, True)
    test_output112 = pack_float32_to_4bit_cpp_based(npu8, True)
    test_output113 = pack_float32_to_4bit(npu8, True)
    _validate_results(test_output111, test_output112)
    _validate_results(test_output111, test_output113)

    # test-case-12: Signed=False, input = numpy array of dtype uint8, size = odd
    test_output121 = pack_float32_to_4bit_optimized(npu8_odd, False)
    test_output122 = pack_float32_to_4bit_cpp_based(npu8_odd, False)
    test_output123 = pack_float32_to_4bit(npu8_odd, False)
    _validate_results(test_output121, test_output122)
    _validate_results(test_output121, test_output123)

    np64 = np.asarray(input_pattern, dtype=np.float64)
    np64_odd = np.asarray(input_pattern[:-1], dtype=np.float64)

    # test-case-13: Signed=True, input = numpy array of dtype float64, size = even
    test_output131 = pack_float32_to_4bit_optimized(np64, True)
    test_output132 = pack_float32_to_4bit_cpp_based(np64, True)
    test_output133 = pack_float32_to_4bit(np64, True)
    _validate_results(test_output131, test_output132)
    _validate_results(test_output131, test_output133)

    # test-case-14: Signed=False, input = numpy array of dtype float64, size = odd
    test_output141 = pack_float32_to_4bit_optimized(np64_odd, False)
    test_output142 = pack_float32_to_4bit_cpp_based(np64_odd, False)
    test_output143 = pack_float32_to_4bit(np64_odd, False)
    _validate_results(test_output141, test_output142)
    _validate_results(test_output141, test_output143)

    npf16 = np.asarray(input_pattern, dtype=np.float16)
    npf16_odd = np.asarray(input_pattern[:-1], dtype=np.float16)

    # test-case-15: Signed=True, input = numpy array of dtype float16, size = even
    test_output151 = pack_float32_to_4bit_optimized(npf16, True)
    test_output152 = pack_float32_to_4bit_cpp_based(npf16, True)
    test_output153 = pack_float32_to_4bit(npf16, True)
    _validate_results(test_output151, test_output152)
    _validate_results(test_output151, test_output153)

    # test-case-16: Signed=False, input = numpy array of dtype float16, size = odd
    test_output161 = pack_float32_to_4bit_optimized(npf16_odd, False)
    test_output162 = pack_float32_to_4bit_cpp_based(npf16_odd, False)
    test_output163 = pack_float32_to_4bit(npf16_odd, False)
    _validate_results(test_output161, test_output162)
    _validate_results(test_output161, test_output163)

    input_pattern_int4_boundary = [-8, 0, 7, 0, -8, 7]
    np_int4_boundary = np.asarray(input_pattern_int4_boundary, dtype=np.int8)

    # test-case-17: Signed=True, input = numpy array of dtype int8, size = even,
    #               Input values are boundary values in int4 range
    test_output171 = pack_float32_to_4bit_optimized(np_int4_boundary, True)
    test_output172 = pack_float32_to_4bit_cpp_based(np_int4_boundary, True)
    test_output173 = pack_float32_to_4bit(np_int4_boundary, True)
    _validate_results(test_output171, test_output172)
    _validate_results(test_output171, test_output173)

    input_pattern_uint4_boundary = [15, 0, 7, 0]
    np_uint4_boundary = np.asarray(input_pattern_uint4_boundary, dtype=np.uint8)

    # test-case-18: Signed=False, input = numpy array of dtype uint8, size = even,
    #               Input values are boundary values in uint4 range
    test_output181 = pack_float32_to_4bit_optimized(np_uint4_boundary, False)
    test_output182 = pack_float32_to_4bit_cpp_based(np_uint4_boundary, False)
    test_output183 = pack_float32_to_4bit(np_uint4_boundary, False)
    _validate_results(test_output181, test_output182)
    _validate_results(test_output181, test_output183)
