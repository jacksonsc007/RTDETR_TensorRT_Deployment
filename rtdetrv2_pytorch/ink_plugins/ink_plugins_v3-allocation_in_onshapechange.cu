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

//!
//! sampleNonZeroPlugin.cpp
//! This file contains a sample demonstrating a plugin for NonZero.
//! It can be run with the following command line:
//! Command: ./sample_non_zero_plugin [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//!



#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include <cstdint>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>
#include <vector>
#include <memory>
#include <cstring>

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

constexpr size_t alignSize(size_t size, size_t alignment = 16) {
    return (size + alignment - 1) & ~(alignment - 1);
}

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
                    "\e[32m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
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
        //     printf("\e[32mRows:%d Cols:%d [%d, %d] =  %f\e[m \n", n_rows, n_cols, i, j, acc);
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
                    "\e[32m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
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
                    "\e[32m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
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
                    "\e[32m[DEBUG]\e[m activation_scale: %3.4f, activation_offset: %4d activation: %4d;"
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
class fused_attn_offset_prediction : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
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
    // std::string mNamespace = "";
    // std::string mNamespace;
    // NOTE: plugin version, type and name are wrapped in methods
    
    // The followings are pointers to GPU, where the computation occurs
    // There pointers are initilizaed in configure_plugin()
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

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
    int32_t tactic = -1;

public:
    fused_attn_offset_prediction(fused_attn_offset_prediction const& p) = default;

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
        printf("\e[32m[Plugin::initializer]\e[m \n");
        #endif
        initFieldsToSerialize();
    }

    void initFieldsToSerialize()
    {
        mDataToSerialize.clear();
        // mDataToSerialize.emplace_back(PluginField("rowOrder", &mRowOrder, PluginFieldType::kINT32, 1));
        mDataToSerialize.emplace_back(PluginField("input_scale", input_scale.data(), PluginFieldType::kFLOAT32, 1));
        mDataToSerialize.emplace_back(PluginField("input_offset", input_offset.data(), PluginFieldType::kINT8, 1));
        mDataToSerialize.emplace_back(PluginField("sp_weight_scale", sp_weight_scale.data(), PluginFieldType::kFLOAT32, 192));
        mDataToSerialize.emplace_back(PluginField("sp_weight_offset", sp_weight_offset.data(), PluginFieldType::kINT8, 192));
        mDataToSerialize.emplace_back(PluginField("attn_weight_scale", attn_weight_scale.data(), PluginFieldType::kFLOAT32, 96));
        mDataToSerialize.emplace_back(PluginField("attn_weight_offset", attn_weight_offset.data(), PluginFieldType::kINT8, 96));
        mDataToSerialize.emplace_back(PluginField("int8_sp_weight", int8_sp_weight.data(), PluginFieldType::kINT8, 192 * 256));
        mDataToSerialize.emplace_back(PluginField("int8_attn_weight", int8_attn_weight.data(), PluginFieldType::kINT8, 96 * 256));
        mDataToSerialize.emplace_back(PluginField("sp_fp32_bias", sp_fp32_bias.data(), PluginFieldType::kFLOAT32, 192));
        mDataToSerialize.emplace_back(PluginField("attn_fp32_bias", attn_fp32_bias.data(), PluginFieldType::kFLOAT32, 96));
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
    }

    // IPluginV3 methods

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        try
        {
            if (type == PluginCapabilityType::kBUILD)
            {
                return static_cast<IPluginV3OneBuild*>(this);
            }
            if (type == PluginCapabilityType::kRUNTIME)
            {
                return static_cast<IPluginV3OneRuntime*>(this);
            }
            ASSERT(type == PluginCapabilityType::kCORE);
            return static_cast<IPluginV3OneCore*>(this);
        }
        catch (std::exception const& e)
        {
            std::cout << e.what() << std::endl;
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override
    {
        auto clone = std::make_unique<fused_attn_offset_prediction>(*this);
        clone->initFieldsToSerialize();
        return clone.release();
    }

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return "1";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    // IPluginV3OneBuild methods
    int32_t getNbOutputs() const noexcept override
    {
        return 2;
    }
    
    int32_t getNbTactics() noexcept
    {
        // printf("\e[32m[Seting number tactics to 1]\e[m \n");
        return 1;
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
    {
        if (nbTactics != 1)
        {
            std::abort();
        }
        tactics[0] = 1;
        // std::abort();
        return 0;
    }

    int32_t setTactic(int32_t tactic) noexcept
    {
        // printf("\e[32m[get tactic:%d]\e[m \n", tactic);
        this->tactic = tactic;
        return 0;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::configurePlugin]\e[m \n");
        #endif
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        bool typeOk{false};
        if (pos == 0)
        {
            typeOk = inOut[0].desc.type == DataType::kINT8 ;
        }
        else if (pos == 1)
        {
            typeOk = inOut[1].desc.type == DataType::kFLOAT;
        }
        else // pos == 2
        {
            typeOk = inOut[2].desc.type == DataType::kFLOAT;
        }

        // inputs and outputs must observe KLINEAR format (NCHW)
        return inOut[pos].desc.format == PluginFormat::kLINEAR && typeOk;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        outputTypes[0] = DataType::kFLOAT;
        outputTypes[1] = DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        // The input tensor must be 3-D
        if (inputs[0].nbDims != 3)
        {
            return -1;
        }
        if (nbInputs != 1 && nbOutputs != 2)
        {
            return -1;
        }

        outputs[0] = inputs[0];
        outputs[1] = inputs[0];
        
        int32_t nbDims = inputs[0].nbDims;
        outputs[0].d[nbDims - 1] = exprBuilder.constant(96);  // attn output
        outputs[1].d[nbDims - 1] = exprBuilder.constant(192); //  sp output
        return 0;
    }

    // IPluginV3OneRuntime methods
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
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


        // printf("\e[31m[enqueue]\e[m \n");
        // use 2-d block
        const int32_t batchsize = inputDesc[0].dims.d[0];
        const int32_t num_queries = inputDesc[0].dims.d[1];
        const int32_t n_channels = inputDesc[0].dims.d[2];
        const int32_t blocksize = 32;
        const int32_t output_col = 96 + 192;
        dim3 blockDim(blocksize, blocksize, 1);
        dim3 gridDim(div_round_up(output_col, blocksize), div_round_up(num_queries, blocksize), 1);
        // NOTE: Despite the input type is set as float (because we intend to avoid TensorRT treating this as quantization node), we recast it to int8_t.
        int8_t const * input_int8_gpu = static_cast<int8_t const*>(inputs[0]);
        // float const * input_fp32_gpu = static_cast<float const*>(inputs[0]);
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
        float * input_fp_cpu = new float[batchsize * num_queries * n_channels];
        ASSERT(
            !cudaMemcpy((void *)input_fp_cpu, inputs[0], batchsize * num_queries * n_channels * sizeof(float), cudaMemcpyDeviceToHost)
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
        // std::ofstream output_file("plugin_input.bin", std::ios::binary);

        // if (!output_file)
        // {
        //     std::cerr << "Failed to open file" << std::endl;
        //     std::abort();
        // }
        // output_file.write(reinterpret_cast<char *>(input_int8_cpu), batchsize * num_queries * n_channels * sizeof(int8_t));
        // output_file.close();
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
        // copy from fused_output to outputs
        ASSERT(
            !cudaMemcpy(outputs[0], fused_output_ptr_d->mPtr, batchsize * num_queries * 96 * sizeof(float), cudaMemcpyDeviceToDevice)
        );
        ASSERT(
            !cudaMemcpy(outputs[1], fused_output_ptr_d->mPtr + batchsize * num_queries * 96, batchsize * num_queries * 192 * sizeof(float), cudaMemcpyDeviceToDevice)
        );
        
        
        #ifdef DEBUG_KERNEL
        printf("%p\n", input_scale_ptr_d->mPtr);
        printf("%p\n", fused_output_ptr_d->mPtr);
        ASSERT(
            !cudaMemcpy(outputs[0], fused_output_ptr_d->mPtr, batchsize * num_queries * 96 * sizeof(float), cudaMemcpyDeviceToDevice)
        );
        ASSERT(
            !cudaMemcpy(outputs[1], fused_output_ptr_d->mPtr, batchsize * num_queries * 192 * sizeof(float), cudaMemcpyDeviceToDevice)
        );
        #endif
        

        // print output
// #define DEBUG_OUTPUT
#ifdef DEBUG_OUTPUT
        float *attn_output_cpu = new float[batchsize * num_queries * 96];
        float *attn_output_cpu_2 = new float[batchsize * num_queries * 96];
        ASSERT(
            !cudaMemcpy((void *)attn_output_cpu, outputs[0], batchsize * num_queries * 96 * sizeof(float), cudaMemcpyDeviceToHost)
        );
        ASSERT(
            !cudaMemcpy((void *)attn_output_cpu_2, fused_output_ptr_d->mPtr, batchsize * num_queries * 96 * sizeof(float), cudaMemcpyDeviceToHost)
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
                    if (value > max_attn) max_attn = value;
                    if (value < min_attn) min_attn = value;
                    // if (i == 0)
                    printf("%d, %d, %d: %8.4f\n", b, i, j, value);
                }
        }
        printf("max: %4.4f min: %4.4f sum:%4.4f\n", max_attn, min_attn, sum_attn);
#endif

        return 0;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::onShapeChange]\e[m \n");
        #endif

        // get input size and allocate GPU buffer
        int32_t batchsize  = in[0].dims.d[0];
        const int32_t nDims = in[0].dims.nbDims;
        const int32_t num_queries = in[0].dims.d[nDims - 2];
        const int32_t feature_dim = in[0].dims.d[nDims - 1];
        const int32_t attn_dim = 96;
        const int32_t sp_dim = 192;
        #ifdef DEBUG
        // printf("\e[32m[INFO]\e[m configurePlugin ... \n");
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

        return 0;
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::getFieldsToSerialize]\e[m \n");
        #endif
        return &mFCToSerialize;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        #ifdef DEBUG_PLUGIN
        printf("\e[32m[plugin::getWorkspacesize]\e[m \n");
        #endif
        // TODO: likely causing issues
        // return sizeof(int32_t);
        return 1024 * 1024 * 1024;
    }

};

class fused_attn_offset_prediction_Creator : public nvinfer1::IPluginCreatorV3One
{
private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace = PLUGIN_NAMESPACE;
    // std::string mNamespace = "";


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

    char const* getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return "1";
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override
    {

        try
        {
        #ifdef DEBUG_PLUGIN
            printf("\e[32m[Creator::CreatePlugin]\e[m \n");
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

    char const* getPluginNamespace() const noexcept override
    {
        // return "";
        return mNamespace.c_str();

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

IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
     nbCreators = 1;
    // static ROIAlignV3PluginCreator sRoiAlignCreator;
    static fused_attn_offset_prediction_Creator sFusedAttnOffsetPredictionCreator;
    static IPluginCreatorInterface* const kPLUGIN_CREATOR_LIST[] = {&sFusedAttnOffsetPredictionCreator};
    return kPLUGIN_CREATOR_LIST;   
}
} // extern "C"