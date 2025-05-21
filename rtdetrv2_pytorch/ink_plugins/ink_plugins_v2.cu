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

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimePlugin.h"

// #include <__clang_cuda_builtin_vars.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

using namespace nvinfer1;

/*
The following are helper functions.
*/
static void caughtError(std::exception const& e)
{
    std::cout << e.what() << std::endl;
}

// Write values into buffer
template <typename T>
void write(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(char const*& buffer)
{
    T val{};
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}

#define ASSERT(condition)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::cout << "Assertion failure: " << #condition << std::endl;                                             \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define div_round_up(a, b) (((a) + (b) - 1) / (b))

template <typename Dtype>
struct CudaBind
{
    size_t mSize;
    Dtype* mPtr;

    CudaBind(size_t size)
    {
        mSize = size;
        ASSERT(!cudaMalloc((void**) &mPtr, sizeof(Dtype) * mSize));
    }

    ~CudaBind()
    {
        if (mPtr != nullptr)
        {
            ASSERT(!cudaFree(mPtr));
            mPtr = nullptr;
        }
    }
};

static int64_t volume(Dims const& dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}


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


/*
Naive CUDA kernel to perform fused attention and offset prediction
*/
// NOTE: Do pay attention to the memory layout for the fused output
__global__ void fused_attn_offset_prediction_kernel_fp32_mul(
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
    float * fused_output
)
{
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x; // column idx
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y; // row idx

    float s_a = *activation_scale;
    int8_t a_offset = *activation_offset;
    float acc = 0.0f;
    
    // attn result
    if (i < n_rows && 0 < j  && j < 96)
    {
        float w_bias = fused_weight_bias(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t w_offset = fused_weight_offset(0, j, 1);
            float s_w = fused_weight_scale(0, j, 1);
            float dq_w = (w - w_offset) * s_w;

            int8_t a = activation(i, p, nC);
            float dq_a = (a - a_offset) * s_a;

            acc += dq_w * dq_a;
            #ifdef DEBUG_KERNEL
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %3.4f "
                    "dq_w: %8.4f dq_a: %8.4f\n", 
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w, w_bias, n_rows, n_cols, nC, acc,
                    dq_w, dq_a
                );
            #endif
            
        }
        acc += w_bias;
        fused_output(i, j, 96) = acc;
        // if (i == 0 )
        //     printf("\e[31mRows:%d Cols:%d [%d, %d] =  %f\e[m \n", n_rows, n_cols, i, j, acc);
    }
    else if (i < n_rows && j >= 96 && j < n_cols)
    {
        fused_output += n_rows * 96;
        float w_bias = fused_weight_bias(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t w_offset = fused_weight_offset(0, j, 1);
            float s_w = fused_weight_scale(0, j, 1);
            float dq_w = (w - w_offset) * s_w;

            int8_t a = activation(i, p, nC);
            float dq_a = (a - a_offset) * s_a;

            acc += dq_w * dq_a;
            #ifdef DEBUG_KERNEL
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %3.4f "
                    "dq_w: %8.4f dq_a: %8.4f\n", 
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w, w_bias, n_rows, n_cols, nC, acc,
                    dq_w, dq_a
                );
            #endif
            
        }
        acc += w_bias;
        fused_output(i, j - 96, 192) = acc;
        
    }
}

__global__ void fused_attn_offset_prediction_kernel_int8_mul(
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
    float * fused_output
)
{
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x; // column idx
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y; // row idx

    float s_a = *activation_scale;
    int8_t a_offset = *activation_offset;
    int32_t acc_i32 = 0;
    float acc_float = 0.0f;
    
    // attn result
    if (i < n_rows && 0 < j  && j < 96)
    {
        float s_w = fused_weight_scale(0, j, 1);
        float w_bias = fused_weight_bias(0, j, 1);
        int8_t w_offset = fused_weight_offset(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t a = activation(i, p, nC);

            int16_t i8_prod = (w - w_offset) * (a - a_offset);
            acc_i32 += i8_prod;
            #ifdef DEBUG_KERNEL
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %3.4f "
                    "dq_w: %8.4f dq_a: %8.4f\n", 
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w, w_bias, n_rows, n_cols, nC, acc,
                    dq_w, dq_a
                );
            #endif
            
        }
        acc_float = (float)acc_i32 * (s_a * s_w);
        acc_float += w_bias;
        fused_output(i, j, 96) = acc_float;
    }
    else if (i < n_rows && j >= 96 && j < n_cols)
    {
        fused_output += n_rows * 96;
        float s_w = fused_weight_scale(0, j, 1);
        float w_bias = fused_weight_bias(0, j, 1);
        for (int p = 0; p < nC; ++p)
        {
            int8_t w = fused_weight(p, j, nC);
            int8_t w_offset = fused_weight_offset(0, j, 1);
            int8_t a = activation(i, p, nC);

            int16_t i8_prod = (w- w_offset) * (a - a_offset);
            acc_i32 += i8_prod;
            #ifdef DEBUG_KERNEL
            if (i == 0 && j == 0)
                printf(
                    "\e[31m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
                    "fused_weight_scale: %3.4f, fused_weight_offset: %4d, fused_weight: %4d ;"
                    "fused_weight_bias: %3.4f, n_rows: %4d, n_cols: %4d, nC: %4d, acc: %3.4f "
                    "dq_w: %8.4f dq_a: %8.4f\n", 
                    s_a, (int32_t) a_offset, (int32_t)a, 
                    s_w, (int32_t) w_offset, (int32_t) w, w_bias, n_rows, n_cols, nC, acc,
                    dq_w, dq_a
                );
            #endif
            
        }
        acc_float = (float)acc_i32 * (s_a * s_w);
        acc_float += w_bias;
        fused_output(i, j - 96, 192) = acc_float;
        
    }



}

#define PLUGIN_NAME "fused_attn_offset_prediction"
#define PLUGIN_NAMESPACE "ink_plugins"
// #define PLUGIN_NAMESPACE ""
class fused_attn_offset_prediction : public nvinfer1::IPluginV2DynamicExt
{
private:
    // Ink: We put attributes of our custom nodes here
    std::vector<float> input_scale{};
    std::vector<int8_t> input_offset{};
    std::vector<float> sp_weight_scale{};
    std::vector<int8_t> sp_weight_offset{};
    std::vector<float> attn_weight_scale{};
    std::vector<int8_t> attn_weight_offset{};
    std::vector<int8_t> int8_sp_weight{};
    std::vector<int8_t> int8_attn_weight{};
    std::vector<float> sp_fp32_bias{};
    std::vector<float> attn_fp32_bias{};
    std::string mNamespace = PLUGIN_NAMESPACE;
    // std::string mNamespace;
    // NOTE: plugin version, type and name are wrapped in methods
    
    // The followings are pointers to GPU, where the computation occurs
    // There pointers are initilizaed in configurePlugin()
    // TODO: can we omit {}?
    std::shared_ptr<CudaBind<float>> input_scale_ptr_d{};
    std::shared_ptr<CudaBind<int8_t>> input_offset_ptr_d{};
    std::shared_ptr<CudaBind<float>> sp_weight_scale_ptr_d{};
    std::shared_ptr<CudaBind<int8_t>> sp_weight_offset_ptr_d{};
    std::shared_ptr<CudaBind<float>> attn_weight_scale_ptr_d{};
    std::shared_ptr<CudaBind<int8_t>> attn_weight_offset_ptr_d{};
    std::shared_ptr<CudaBind<int8_t>> int8_sp_weight_ptr_d{};
    std::shared_ptr<CudaBind<int8_t>> int8_attn_weight_ptr_d{};
    std::shared_ptr<CudaBind<float>> sp_fp32_bias_ptr_d{};
    std::shared_ptr<CudaBind<float>> attn_fp32_bias_ptr_d{};
    // use fused attributes
    std::shared_ptr<CudaBind<float>> fused_weight_scale_ptr_d{};
    std::shared_ptr<CudaBind<int8_t>> fused_weight_offset_ptr_d{};
    std::shared_ptr<CudaBind<int8_t>> fused_weight_ptr_d{};
    std::shared_ptr<CudaBind<float>> fused_weight_bias_ptr_d{};
    // fused_output
    std::shared_ptr<CudaBind<float>> fused_output_ptr_d{};
public:
    fused_attn_offset_prediction() = default;

    fused_attn_offset_prediction(
        std::vector<float> input_scale,
        std::vector<int8_t> input_offset,
        std::vector<float> sp_weight_scale,
        std::vector<int8_t> sp_weight_offset,
        std::vector<float> attn_weight_scale,
        std::vector<int8_t> attn_weight_offset,
        std::vector<int8_t> int8_sp_weight,
        std::vector<int8_t> int8_attn_weight,
        std::vector<float> sp_fp32_bias,
        std::vector<float> attn_fp32_bias
    )
        : input_scale(input_scale),
          input_offset(input_offset),
          sp_weight_scale(sp_weight_scale),
          sp_weight_offset(sp_weight_offset),
          attn_weight_scale(attn_weight_scale),
          attn_weight_offset(attn_weight_offset),
          int8_sp_weight(int8_sp_weight),
          int8_attn_weight(int8_attn_weight),
          sp_fp32_bias(sp_fp32_bias),
          attn_fp32_bias(attn_fp32_bias)
    {
        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::initializer function]\e[m \n");
        #endif
    }

    fused_attn_offset_prediction(fused_attn_offset_prediction const& p) = default;

    fused_attn_offset_prediction(void const* serialData, size_t length)
    {
        ASSERT(serialData != nullptr);

        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::de-serilization]\e[m \n");
        #endif
        char const* d = static_cast<char const*>(serialData);
        char const* a = d;

        int32_t size;
        size = read<int32_t>(d);
        input_scale.resize(size);
        for (int i = 0; i < size; ++i)
        {
            input_scale[i] = read<float>(d);
        }
        size = read<int32_t>(d);
        input_offset.resize(size);
        for (int i = 0; i < size; ++i)
        {
            input_offset[i] = read<int8_t>(d);
        }

        size = read<int32_t>(d);
        sp_weight_scale.resize(size);
        for (int i = 0; i < size; ++i)
        {
            sp_weight_scale[i] = read<float>(d);
        }

        size = read<int32_t>(d);
        sp_weight_offset.resize(size);
        for (int i = 0; i < size; ++i)
        {
            sp_weight_offset[i] = read<int8_t>(d);
        }

        size = read<int32_t>(d);
        attn_weight_scale.resize(size);
        for (int i = 0; i < size; ++i)
        {
            attn_weight_scale[i] = read<float>(d);
        }

        size = read<int32_t>(d);
        attn_weight_offset.resize(size);
        for (int i = 0; i < size; ++i)
        {
            attn_weight_offset[i] = read<int8_t>(d);
        }

        size = read<int32_t>(d);
        int8_sp_weight.resize(size);
        for (int i = 0; i < size; ++i)
        {
            int8_sp_weight[i] = read<int8_t>(d);
        }

        size = read<int32_t>(d);
        int8_attn_weight.resize(size);
        for (int i = 0; i < size; ++i)
        {
            int8_attn_weight[i] = read<int8_t>(d);
        }

        size = read<int32_t>(d);
        sp_fp32_bias.resize(size);
        for (int i = 0; i < size; ++i)
        {
            sp_fp32_bias[i] = read<float>(d);
        }

        size = read<int32_t>(d);
        attn_fp32_bias.resize(size);
        for (int i = 0; i < size; ++i)
        {
            attn_fp32_bias[i] = read<float>(d);
        }

        ASSERT(d == a + length);
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 2;
    }

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
    {
        PluginTensorDesc const& desc = inOut[pos];
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }

        // first input should be float16 or float32
        if (pos == 0)
        {
            return (inOut[pos].type == nvinfer1::DataType::kINT8);
        }

        // output should have the same type float
        if (pos == 1 || pos == 2)
        {
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT);
        }

        return false;
    }

    void configureWithFormat(nvinfer1::Dims const*, int32_t, nvinfer1::Dims const*, int32_t, nvinfer1::DataType type,
        nvinfer1::PluginFormat floatFormat, int32_t) noexcept override
    {
    }

    int32_t initialize() noexcept override
    {
        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::initialize method]\e[m \n");
        #endif
        return 0;
    }

    void terminate() noexcept override
    {
        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::terminate]\e[m \n");
        #endif
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept
    {
        /*
        In this method, we define the cuda kernel that actually does the computation.
        inputs: int8, (1, 300, 256)
        inputs_scale: float, (1, )
        inputs_offset: int8, (1, )
        int8_attn_weight: int8, (96, 256)
        attn_weight_scale: float, (96,)
        attn_weight_offset: int8, (96,)

        int8_sp_weight: int8, (192, 256)
        sp_weight_scale: float, (192,)
        sp_weight_offset: int8, (192,)
        
        fused_int8_weight: int8, (288, 256)
        fused_fp32_scale: float, (288,)
        fused_int8_offset: int8, (288,)
        fused_fp32_bias: float, (288,)
        
        */
        // 1. prepare inputs and parameters
        
        // use 2-d block
        // #define DEBUG_PLUGIN
        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::enqueue]\e[m \n");
        #endif
        const int32_t batchsize = inputDesc[0].dims.d[0];
        const int32_t num_queries = inputDesc[0].dims.d[1];
        const int32_t n_channels = inputDesc[0].dims.d[2];
        const int32_t blocksize = 32;
        const int32_t output_col = 96 + 192;
        dim3 blockDim(blocksize, blocksize, 1);
        dim3 gridDim(div_round_up(output_col, blocksize), div_round_up(num_queries, blocksize), 1);
        // print attn_weight_scale and sp_weight_scale
        // printf("\e[32m[INFO]\e[m enqueue: batchsize: %d, num_queries: %d, n_channels: %d, blocksize: %d, output_col: %d\n", 
        //     batchsize, num_queries, n_channels, blocksize, output_col
        // );
        // print grid and block info
        // printf("\e[32m[INFO]\e[m enqueue: gridDim: (%d, %d, %d), blockDim: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
        // printf("\e[32m[INFO]\e[m enqueue: attn_weight_scale: %f, sp_weight_scale: %f\n", attn_weight_scale[0], sp_weight_scale[0]);
        int8_t const * input_int8_gpu = static_cast<int8_t const*>(inputs[0]);
        // #define DEBUG_INPUT
        #ifdef DEBUG_INPUT
            cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100 * 1024 * 1024);
            // print input for debugging
            // first move from gpu to cpu
            // TODO: use smart pointer
            int8_t * input_int8_cpu = new int8_t[batchsize * num_queries * n_channels];
            ASSERT(
                !cudaMemcpy((void *)input_int8_cpu, inputs[0], batchsize * num_queries * n_channels * sizeof(int8_t), cudaMemcpyDeviceToHost)
            );

            printf("\e[32m[Input]\e[m \n");
            int32_t max = 0;
            int32_t min = 400;
            int64_t sum = 0;
            for (int b = 0; b < batchsize; b++)
            {
                printf("batch: %d\n", b);
                for (int i = 0; i < num_queries; i++)
                for (int j = 0 ; j < n_channels; j++)
                {
                    int32_t value = (int32_t)(input_int8_cpu[b * num_queries * n_channels + i * n_channels + j]);
                    sum += (int64_t) value;
                    if (value > max) max = value;
                    if (value < min) min = value;
                    if (i == 0)
                        printf("%d, %d, %d: %d\n", b, i, j, value);
                }
            }
            printf("max: %d min: %d sum:%ld\n", max, min, sum);
            // save input_int8_cpu to file
            std::ofstream output_file("plugin_input.bin", std::ios::binary);

            if (!output_file)
            {
                std::cerr << "Failed to open file" << std::endl;
                std::abort();
            }
            output_file.write(reinterpret_cast<char *>(input_int8_cpu), batchsize * num_queries * n_channels * sizeof(int8_t));
            output_file.close();
        #endif




        fused_attn_offset_prediction_kernel_int8_mul<<<gridDim, blockDim, 0, stream>>>(
            input_int8_gpu,
            input_scale_ptr_d->mPtr,
            input_offset_ptr_d->mPtr,
            fused_weight_scale_ptr_d->mPtr,
            fused_weight_offset_ptr_d->mPtr,
            fused_weight_ptr_d->mPtr,
            fused_weight_bias_ptr_d->mPtr,
            num_queries,
            output_col,
            n_channels,
            fused_output_ptr_d->mPtr
        );
        cudaDeviceSynchronize();
        // copy from fused_output to outputs
        ASSERT(
            !cudaMemcpy(outputs[0], fused_output_ptr_d->mPtr, batchsize * num_queries * 96 * sizeof(float), cudaMemcpyDeviceToDevice)
        );
        ASSERT(
            !cudaMemcpy(outputs[1], fused_output_ptr_d->mPtr + batchsize * num_queries * 96, batchsize * num_queries * 192 * sizeof(float), cudaMemcpyDeviceToDevice)
        );
        // print output
        #ifdef DEBUG_OUTPUT
        float *attn_output_cpu = new float[batchsize * num_queries * 96];
        ASSERT(
            !cudaMemcpy((void *)attn_output_cpu, outputs[0], batchsize * num_queries * 96 * sizeof(float), cudaMemcpyDeviceToHost)
        );
        printf("\e[32m[Attn output]\e[m \n");
        float max_attn = 0;
        float min_attn = 400;
        float sum_attn = 0;
        for (int b = 0; b < batchsize; b++)
        {
            printf("batch: %d\n", b);
            for (int i = 0; i < num_queries; i++)
            for (int j = 0 ; j < 96; j++)
            {
                float value = (attn_output_cpu[b * num_queries * n_channels + i * 96 + j]);
                sum_attn += (float) value;
                if (value > max) max = value;
                if (value < min) min = value;
                // if (i == 0)
                    printf("%d, %d, %d: %8.4f\n", b, i, j, value);
            }
        }
        printf("max: %4.4f min: %4.4f sum:%4.4f\n", max_attn, min_attn, sum_attn);
        #endif
        
        return 0;
    }

    size_t getSerializationSize() const noexcept override
    {
        size_t total_size =  0;
        // NOTE: We need additional 1 to account for the size of each attribute
        total_size += input_scale.size() * sizeof(float);
        total_size += 1 * sizeof(int32_t);
        total_size += input_offset.size() * sizeof(int8_t);
        total_size += 1 * sizeof(int32_t);
        total_size += sp_weight_scale.size() * sizeof(float);
        total_size += 1 * sizeof(int32_t);
        total_size += sp_weight_offset.size() * sizeof(int8_t);
        total_size += 1 * sizeof(int32_t);
        total_size += attn_weight_scale.size() * sizeof(float);
        total_size += 1 * sizeof(int32_t);
        total_size += attn_weight_offset.size() * sizeof(int8_t);
        total_size += 1 * sizeof(int32_t);
        total_size += int8_sp_weight.size() * sizeof(int8_t);
        total_size += 1 * sizeof(int32_t);
        total_size += int8_attn_weight.size() * sizeof(int8_t);
        total_size += 1 * sizeof(int32_t);
        total_size += sp_fp32_bias.size() * sizeof(float);
        total_size += 1 * sizeof(int32_t);
        total_size += attn_fp32_bias.size() * sizeof(float);
        total_size += 1 * sizeof(int32_t);
        return total_size;
    }

    void serialize(void* buffer) const noexcept override
    {
        ASSERT(buffer != nullptr);
        char* d = static_cast<char*>(buffer);
        char* a = d;
        // save the size for later deserialization
        write(d, static_cast<int32_t>(input_scale.size()));
        for (int i = 0; i < input_scale.size(); ++i)
        {
            write(d, input_scale[i]);
        }

        write(d, static_cast<int32_t>(input_offset.size()));
        for (int i = 0; i < input_offset.size(); ++i)
        {
            write(d, input_offset[i]);
        }

        write(d, static_cast<int32_t>(sp_weight_scale.size()));
        for (int i = 0; i < sp_weight_scale.size(); ++i)
        {
            write(d, sp_weight_scale[i]);
        }

        write(d, static_cast<int32_t>(sp_weight_offset.size()));
        for (int i = 0; i < sp_weight_offset.size(); ++i)
        {
            write(d, sp_weight_offset[i]);
        }

        write(d, static_cast<int32_t>(attn_weight_scale.size()));
        for (int i = 0; i < attn_weight_scale.size(); ++i)
        {
            write(d, attn_weight_scale[i]);
        }

        write(d, static_cast<int32_t>(attn_weight_offset.size()));
        for (int i = 0; i < attn_weight_offset.size(); ++i)
        {
            write(d, attn_weight_offset[i]);
        }

        write(d, static_cast<int32_t>(int8_sp_weight.size()));
        for (int i = 0; i < int8_sp_weight.size(); ++i)
        {
            write(d, int8_sp_weight[i]);
        }

        write(d, static_cast<int32_t>(int8_attn_weight.size()));
        for (int i = 0; i < int8_attn_weight.size(); ++i)
        {
            write(d, int8_attn_weight[i]);
        }

        write(d, static_cast<int32_t>(sp_fp32_bias.size()));
        for (int i = 0; i < sp_fp32_bias.size(); ++i)
        {
            write(d, sp_fp32_bias[i]);
        }

        write(d, static_cast<int32_t>(attn_fp32_bias.size()));
        for (int i = 0; i < attn_fp32_bias.size(); ++i)
        {
            write(d, attn_fp32_bias[i]);
        }

        int32_t offset = (a + getSerializationSize() - d);
        if (offset != 0)
        {

            std::cout << "mismatch: " << offset << std::endl;
            std::abort();
        }

        ASSERT(d == a + getSerializationSize());
    }

    char const* getPluginType() const noexcept override
    {
        return PLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return "1";
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        // TODO: Memory leak?
        return new fused_attn_offset_prediction(*this);
    }

    void destroy() noexcept override
    {
        delete this;
    }

    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
        // return "ink_plugins";
    }

    DataType getOutputDataType(int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
    {
        if (index == 0) return nvinfer1::DataType::kFLOAT;
        else if (index == 1) return nvinfer1::DataType::kFLOAT;
        else
        {
            std::abort();
        };
    }

    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
    {
        int32_t nbOutDims = inputs[0].nbDims;
        if ( outputIndex ==  0 )
        {
            nvinfer1::DimsExprs outDims_attn{inputs[0]};
            outDims_attn.d[nbOutDims - 1] = exprBuilder.constant(96);
            return outDims_attn;
        }
        else if (outputIndex == 1)
        {
            nvinfer1::DimsExprs outDims_offset{inputs[0]};
            outDims_offset.d[nbOutDims - 1] = exprBuilder.constant(192);
            return outDims_offset;
        }
        else std::abort();
    }

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept
    {
        // get input size and allocate GPU buffer
        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::configurePlugin]\e[m \n");
        #endif
        int32_t batchsize  = in[0].desc.dims.d[0];
        const int32_t nDims = in[0].desc.dims.nbDims;
        const int32_t num_queries = in[0].desc.dims.d[nDims - 2];
        const int32_t feature_dim = in[0].desc.dims.d[nDims - 1];
        const int32_t attn_dim = 96;
        const int32_t sp_dim = 192;
        #ifdef DEBUG
        printf("\e[32m[INFO]\e[m configurePlugin ... \n");
        printf("\e[32m[INFO]\e[m Input size: %d, %d, %d\n", batchsize, num_queries, feature_dim);
        #endif

        // TODO: Understand CudaBind here
        input_scale_ptr_d = std::make_shared<CudaBind<float>>(1);
        input_offset_ptr_d = std::make_shared<CudaBind<int8_t>>(1);
        sp_weight_scale_ptr_d = std::make_shared<CudaBind<float>>(sp_dim);
        sp_weight_offset_ptr_d = std::make_shared<CudaBind<int8_t>>(sp_dim);
        attn_weight_scale_ptr_d = std::make_shared<CudaBind<float>>(attn_dim);
        attn_weight_offset_ptr_d = std::make_shared<CudaBind<int8_t>>(attn_dim);
        int8_sp_weight_ptr_d = std::make_shared<CudaBind<int8_t>>(sp_dim * feature_dim);
        int8_attn_weight_ptr_d = std::make_shared<CudaBind<int8_t>>(attn_dim * feature_dim);
        sp_fp32_bias_ptr_d = std::make_shared<CudaBind<float>>(sp_dim);
        attn_fp32_bias_ptr_d = std::make_shared<CudaBind<float>>(attn_dim);

        // TODO: why not 2-level pointer here?
        ASSERT(
            !cudaMemcpy(input_scale_ptr_d ->mPtr, &input_scale.front(), input_scale.size() * sizeof(float), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(input_offset_ptr_d ->mPtr, &input_offset.front(), input_offset.size() * sizeof(int8_t), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(sp_weight_scale_ptr_d->mPtr, &sp_weight_scale.front(), sp_weight_scale.size() * sizeof(float), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(sp_weight_offset_ptr_d->mPtr, &sp_weight_offset.front(), sp_weight_offset.size() * sizeof(int8_t), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(attn_weight_scale_ptr_d->mPtr, &attn_weight_scale.front(), attn_weight_scale.size() * sizeof(float), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(attn_weight_offset_ptr_d->mPtr, &attn_weight_offset.front(), attn_weight_offset.size() * sizeof(int8_t), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(int8_sp_weight_ptr_d->mPtr, &int8_sp_weight.front(), int8_sp_weight.size() * sizeof(int8_t), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(int8_attn_weight_ptr_d->mPtr, &int8_attn_weight.front(), int8_attn_weight.size() * sizeof(int8_t), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(sp_fp32_bias_ptr_d->mPtr, &sp_fp32_bias.front(), sp_fp32_bias.size() * sizeof(float), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(attn_fp32_bias_ptr_d->mPtr, &attn_fp32_bias.front(), attn_fp32_bias.size() * sizeof(float), cudaMemcpyHostToDevice)
        );
        


        fused_weight_scale_ptr_d = std::make_shared<CudaBind<float>>(attn_dim + sp_dim);
        fused_weight_offset_ptr_d = std::make_shared<CudaBind<int8_t>>(attn_dim + sp_dim);
        fused_weight_ptr_d = std::make_shared<CudaBind<int8_t>>((attn_dim + sp_dim) * feature_dim);
        fused_weight_bias_ptr_d = std::make_shared<CudaBind<float>>(attn_dim + sp_dim);
        fused_output_ptr_d = std::make_shared<CudaBind<float>>(batchsize * num_queries * (attn_dim + sp_dim));
        // print attn_weight_scale and sp_weight_scale
        // printf("\e[32m[INFO]\e[m attn_weight_scale: %f, sp_weight_scale: %f\n", attn_weight_scale[0], sp_weight_scale[0]);
        // std::abort();
        ASSERT(
            !cudaMemcpy(fused_weight_scale_ptr_d->mPtr, &attn_weight_scale.front(), attn_weight_scale.size() * sizeof(float), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(fused_weight_scale_ptr_d->mPtr + attn_weight_scale.size(), &sp_weight_scale.front(), sp_weight_scale.size() * sizeof(float), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(fused_weight_offset_ptr_d->mPtr, &attn_weight_offset.front(), attn_weight_offset.size() * sizeof(int8_t), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(fused_weight_offset_ptr_d->mPtr + attn_weight_offset.size(), &sp_weight_offset.front(), sp_weight_offset.size() * sizeof(int8_t), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(fused_weight_ptr_d->mPtr, &int8_attn_weight.front(), int8_attn_weight.size() * sizeof(int8_t), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(fused_weight_ptr_d->mPtr + int8_attn_weight.size(), &int8_sp_weight.front(), int8_sp_weight.size() * sizeof(int8_t), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(fused_weight_bias_ptr_d->mPtr, &attn_fp32_bias.front(), attn_fp32_bias.size() * sizeof(float), cudaMemcpyHostToDevice)
        );
        ASSERT(
            !cudaMemcpy(fused_weight_bias_ptr_d->mPtr + attn_fp32_bias.size(), &sp_fp32_bias.front(), sp_fp32_bias.size() * sizeof(float), cudaMemcpyHostToDevice)
        );
        
        
    }

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept
    {
        
        // return 1024 * 1024 * 1024;
        return 0;
    }

};

class fused_attn_offset_prediction_Creator : public nvinfer1::IPluginCreator
{
private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace = PLUGIN_NAMESPACE;
    // std::string mNamespace;

public:
    fused_attn_offset_prediction_Creator()
    {
        mPluginAttributes.clear();
        // mPluginAttributes.emplace_back(PluginField("pads", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("input_scale", nullptr, PluginFieldType::kFLOAT32, 1));
        mPluginAttributes.emplace_back(PluginField("input_offset", nullptr, PluginFieldType::kINT8, 1));
        mPluginAttributes.emplace_back(PluginField("sp_weight_scale", nullptr, PluginFieldType::kFLOAT32, 192));
        mPluginAttributes.emplace_back(PluginField("sp_weight_offset", nullptr, PluginFieldType::kINT8, 192));
        mPluginAttributes.emplace_back(PluginField("attn_weight_scale", nullptr, PluginFieldType::kFLOAT32, 96));
        mPluginAttributes.emplace_back(PluginField("attn_weight_offset", nullptr, PluginFieldType::kINT8, 96));
        mPluginAttributes.emplace_back(PluginField("int8_sp_weight", nullptr, PluginFieldType::kINT8, 192 * 256));
        mPluginAttributes.emplace_back(PluginField("int8_attn_weight", nullptr, PluginFieldType::kINT8, 96 * 256));
        mPluginAttributes.emplace_back(PluginField("sp_fp32_bias", nullptr, PluginFieldType::kFLOAT32, 192));
        mPluginAttributes.emplace_back(PluginField("attn_fp32_bias", nullptr, PluginFieldType::kFLOAT32, 96));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    char const* getPluginName() const noexcept
    {
        return PLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept
    {
        return "1";
    }

    PluginFieldCollection const* getFieldNames() noexcept
    {
        return &mFC;
    }

    IPluginV2* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
    {
        try
        {
        #ifdef DEBUG_PLUGIN
            printf("\e[32m[creator::createPlugin]\e[m \n");
        #endif
            std::vector<float> input_scale;
            std::vector<int8_t> input_offset;
            std::vector<float> sp_weight_scale;
            std::vector<int8_t> sp_weight_offset;
            std::vector<float> attn_weight_scale;
            std::vector<int8_t> attn_weight_offset;
            std::vector<int8_t> int8_sp_weight;
            std::vector<int8_t> int8_attn_weight;
            std::vector<float> sp_fp32_bias;
            std::vector<float> attn_fp32_bias;

            for (int32_t i = 0; i < fc->nbFields; i++)
            {
                std::string field_name(fc->fields[i].name);
                if (field_name == "input_scale")
                {
                    input_scale.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<float const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, input_scale.data());
                }
                else if (field_name == "input_offset")
                {
                    input_offset.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<int8_t const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, input_offset.data());
                }
                else if (field_name == "sp_weight_scale")
                {
                    sp_weight_scale.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<float const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, sp_weight_scale.data());
                }
                else if (field_name == "sp_weight_offset")
                {
                    sp_weight_offset.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<int8_t const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, sp_weight_offset.data());
                }
                else if (field_name == "attn_weight_scale")
                {
                    attn_weight_scale.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<float const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, attn_weight_scale.data());
                }
                else if (field_name == "attn_weight_offset")
                {
                    attn_weight_offset.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<int8_t const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, attn_weight_offset.data());
                }
                else if (field_name == "int8_sp_weight")
                {
                    int8_sp_weight.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<int8_t const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, int8_sp_weight.data());
                }
                else if (field_name == "int8_attn_weight")
                {
                    int8_attn_weight.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<int8_t const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, int8_attn_weight.data());
                }
                else if (field_name == "sp_fp32_bias")
                {
                    sp_fp32_bias.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<float const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, sp_fp32_bias.data());
                }
                else if (field_name == "attn_fp32_bias")
                {
                    attn_fp32_bias.resize(fc->fields[i].length);
                    auto const* dataPtr = static_cast<float const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, attn_fp32_bias.data());
                }
            }

            return new fused_attn_offset_prediction(
                input_scale, input_offset, sp_weight_scale, sp_weight_offset, attn_weight_scale, attn_weight_offset,
                int8_sp_weight, int8_attn_weight, sp_fp32_bias, attn_fp32_bias);

        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    IPluginV2* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
    {
        try
        {
            return new fused_attn_offset_prediction(serialData, serialLength);
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    void setPluginNamespace(char const* libNamespace) noexcept
    {
        // std::cout << "setPluginNamespace" << libNamespace << std::endl;
        #ifdef DEBUG
        printf("\e[31m[INFO]\e[m setPluginNamespace %s\n", libNamespace);
        #endif
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept
    {
        return mNamespace.c_str();
        // return "ink_plugins";
    }

};


// NOTE: We do not use this since we intend to register the plugin into custom namespace
REGISTER_TENSORRT_PLUGIN(fused_attn_offset_prediction_Creator);

// NOTE: Responbile for registering the plugin into custom namespace
extern "C" {
bool initLibNvInferPlugins(void* logger, const char* libNamespace) {
    if (libNamespace && std::string(libNamespace) != "ink_plugins") {
        return false;
    }
    // Register your plugin creator(s)
    getPluginRegistry()->registerCreator(*new fused_attn_offset_prediction_Creator(), "ink_plugins");
    // IPluginRegistry::registerCreator(*new fused_attn_offset_prediction_Creator(), "ink_plugins");

    return true;
}

// This part deal with the logic to find custom loggers, which is not necessary so far.
void setLoggerFinder(ILoggerFinder* finder)
{
}
IPluginCreator* const* getCreators(int32_t& nbCreators)
{
     nbCreators = 1;
    // static ROIAlignV3PluginCreator sRoiAlignCreator;
    static fused_attn_offset_prediction_Creator sFusedAttnOffsetPredictionCreator;
    static IPluginCreator* const kPLUGIN_CREATOR_LIST[] = {&sFusedAttnOffsetPredictionCreator};
    return kPLUGIN_CREATOR_LIST;   
}


// NOTE: Alternative way for getCreators
// IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
// {
//      nbCreators = 1;
//     // static ROIAlignV3PluginCreator sRoiAlignCreator;
//     static fused_attn_offset_prediction_Creator sFusedAttnOffsetPredictionCreator;
//     static IPluginCreatorInterface* const kPLUGIN_CREATOR_LIST[] = {&sFusedAttnOffsetPredictionCreator};
//     return kPLUGIN_CREATOR_LIST;   
// }
} // extern "C"