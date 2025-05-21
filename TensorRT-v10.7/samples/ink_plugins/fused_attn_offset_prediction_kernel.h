/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FUSED_ATTN_OFFSET_PREDICTION_KERNEL_H
#define FUSED_ATTN_OFFSET_PREDICTION_KERNEL_H

#include <cuda_runtime_api.h>
// #include <cuda_fp16.h>
#include <cstdint>

#define div_round_up(a, b) (((a) + (b) - 1) / (b))
// column-major weight layout, as we need to transpose
#define fused_weight(i, j, ld) *( fused_weight + (j) * (ld) + (i) ) 
#define fused_weight_scale(i, j, ld) *( fused_weight_scale + (j) * (ld) + (i) )
#define fused_weight_bias(i, j, ld) *( fused_weight_bias + (j) * (ld) + (i) )
#define fused_weight_offset(i, j, ld) *( fused_weight_offset + (j) * (ld) + (i) )
// row-major weight layout
#define activation(i, j, ld)  *( activation + (i) * (ld) + (j) )
#define activation_scale(i, j, ld)  *( activation_scale + (i) * (ld) + (j) )
#define activation_offset(i, j, ld)  *( activation_offset + (i) * (ld) + (j) )

// TODO: Consider that memory layout should we should for fused output? 
// #define fused_output(i, j, ld) *( fused_output + (j) * (ld) + (i) )
#define fused_output(i, j, ld) *( fused_output + (i) * (ld) + (j) )

enum class Tactic : int32_t
{
    int8_mul = 1,
    float_mul = 2
};

void fused_attn_offset_prediction_impl(
    const int32_t batchsize,
    const int32_t num_queries,
    const int32_t n_channels,
    int8_t const * activation,
    float const* activation_scale,
    int8_t const* activation_offset,
    float const* fused_weight_scale,
    int8_t const* fused_weight_offset,
    int8_t const* fused_weight,
    float const* fused_weight_bias,
    int32_t n_rows,
    int32_t n_cols,
    int32_t nC, // number channels 
    float * fused_output,
    cudaStream_t stream,
    Tactic tactic
);

void debug_fused_attn_offset_prediction_impl(
    const int32_t batchsize,
    const int32_t num_queries,
    const int32_t n_channels,
    float const * activation,
    float const* activation_scale,
    int8_t const* activation_offset,
    float const* fused_weight_scale,
    int8_t const* fused_weight_offset,
    int8_t const* fused_weight,
    float const* fused_weight_bias,
    int32_t n_rows,
    int32_t n_cols,
    int32_t nC, // number channels 
    float * fused_output,
    cudaStream_t stream,
    Tactic tactic
);
#endif // FUSED_ATTN_OFFSET_PREDICTION_KERNEL_H
