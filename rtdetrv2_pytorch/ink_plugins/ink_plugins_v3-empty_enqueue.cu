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
        return 0;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {

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