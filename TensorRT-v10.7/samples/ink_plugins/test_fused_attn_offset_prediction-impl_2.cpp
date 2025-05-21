/*
Second successful implementation of fused_attn_offset_prediction plugin.

NOTE:
1. Plugin attributes are allocated to GPU memory in fused_attn_offset_prediction::enqueue.


! run with:
out/ink_plugins_debug --datadir samples/ink_plugins/validation_data/
 */

// Define TRT entrypoints used in common code
#include "NvInferRuntime.h"
#include "safeCommon.h"
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "fused_attn_offset_prediction_kernel.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

std::string const kSAMPLE_NAME = "Ink.fused_attn_offset_prediction_plugin";

// using half = __half;


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
    Tactic tactic;

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
        printf("\e[31m[Init Plugin]\e[m \n");
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
        printf("\e[31m[Seting number tactics to 2]\e[m \n");
        return 2;
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
    {
        if (nbTactics != 2)
        {
            std::abort();
        }
        tactics[0] = 1;
        tactics[1] = 2;
        // std::abort();
        return 0;
    }

    int32_t setTactic(int32_t tactic) noexcept
    {
        printf("\e[31m[get tactic:%d]\e[m \n", tactic);
        if (tactic == 1)
            this->tactic = Tactic::int8_mul;
        else if (tactic == 2)
            this->tactic = Tactic::float_mul;
        else
        {
            std::cout << "Invalid tactic: " << tactic << std::endl;
            return -1;
        }
        return 0;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
/*         printf("\e[31m[configurePlugin]\e[m \n");
        // get input size and allocate GPU buffer
        int32_t batchsize  = in[0].desc.dims.d[0];
        const int32_t nDims = in[0].desc.dims.nbDims;
        const int32_t num_queries = in[0].desc.dims.d[nDims - 2];
        const int32_t feature_dim = in[0].desc.dims.d[nDims - 1];
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
        ); */
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        bool typeOk{false};
        if (pos == 0)
        {
            typeOk = inOut[0].desc.type == DataType::kFLOAT;
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


        printf("\e[31m[enqueue]\e[m \n");
        
        // #1 get input size and allocate GPU buffer
        int32_t batchsize  = inputDesc[0].dims.d[0];
        const int32_t num_queries = inputDesc[0].dims.d[1];
        const int32_t feature_dim = inputDesc[0].dims.d[2];
        const int32_t n_channels = feature_dim;
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


        
        // use 2-d block
        const int32_t blocksize = 32;
        const int32_t output_col = 96 + 192;
        dim3 blockDim(blocksize, blocksize, 1);
        dim3 gridDim(div_round_up(output_col, blocksize), div_round_up(num_queries, blocksize), 1);

        // NOTE: Despite the input type is set as float (because we intend to avoid TensorRT treating this as quantization node), we recast it to int8_t.
        // The following three casting all work.
        // int8_t const * input_int8_gpu = static_cast<int8_t const*>(inputs[0]);
        int8_t const * input_int8_gpu = reinterpret_cast<int8_t const*>(inputs[0]);
        // int8_t const * input_int8_gpu = (int8_t *)(inputs[0]);

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
        if (workspace == nullptr)
        {
            sample::gLogError << "Unsupported: workspace is null" << std::endl;
            return -1;
        }
        fused_attn_offset_prediction_impl(
            batchsize,
            num_queries,
            n_channels,
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
            fused_output_ptr_d->mPtr,
            stream,
            tactic
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
        printf("\e[32m[onShapeChane]\e[m \n");
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFCToSerialize;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
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
            printf("\e[31m[CreatePlugin]\e[m \n");
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
                    if (fc->fields[i].data == nullptr)
                    {
                        std::cout << "int8_sp_weight data is null" << std::endl;
                        return nullptr;
                    }
                    auto const* dataPtr = static_cast<int8_t const*>(fc->fields[i].data);
                    std::copy_n(dataPtr, fc->fields[i].length, int8_sp_weight.data());
                    // std::memcpy(int8_sp_weight.data(), dataPtr, fc->fields[i].length * sizeof(int8_t));

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

namespace
{
// NOTE: This class encapsulates the fields parameters for the plugin
struct FusedAttnSpParams : public samplesCommon::SampleParams
{
    std::vector<float>  input_scale;
    std::vector<int8_t> input_offset;
    std::vector<float> sp_weight_scale;
    std::vector<int8_t> sp_weight_offset;
    std::vector<float> attn_weight_scale;
    std::vector<int8_t> attn_weight_offset;
    std::vector<int8_t> int8_sp_weight;
    std::vector<int8_t> int8_attn_weight;
    std::vector<float> sp_fp32_bias;
    std::vector<float> attn_fp32_bias;
};
} // namespace

//! \brief  The SampleNonZeroPlugin class implements a NonZero plugin
//!
//! \details The plugin is able to output the non-zero indices in row major or column major order
//!
class TestFusedAttnSpPredictionPlugin
{
public:
    TestFusedAttnSpPredictionPlugin(FusedAttnSpParams const& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
        mSeed = static_cast<uint32_t>(time(nullptr));
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    std::string mNamespace{PLUGIN_NAMESPACE}; //!< The namespace for the plugin
    FusedAttnSpParams mParams; //!< The parameters for the sample.

    std::vector<nvinfer1::Dims> mInputDims;  //!< The dimensions of the input to the network.
    std::vector<nvinfer1::Dims> mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    uint32_t mSeed{};

    //!
    //! \brief Creates a TensorRT network and inserts a NonZero plugin
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input and stores the result in a managed buffer
    //!
    bool processInput(samplesCommon::BufferManager const& buffers);

    //!
    //! \brief Verifies the result
    //!
    bool verifyOutput(samplesCommon::BufferManager const& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates a network containing the plugin and builds
//!          the engine that will be used to run the plugin (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool TestFusedAttnSpPredictionPlugin::build()
{
    // enable verbose logging
    sample::gLogger.setReportableSeverity(ILogger::Severity::kVERBOSE);
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger())
    );
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0)
    );
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig()
    );
    if (!config)
    {
        return false;
    }
    // enable verbose logging
    config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
    config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);

    auto pluginCreator = std::make_unique<fused_attn_offset_prediction_Creator>();
    getPluginRegistry()->registerCreator(*pluginCreator.get(), mNamespace.c_str());

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims.emplace_back(network->getInput(0)->getDimensions());

    ASSERT(network->getNbOutputs() == 2);
    mOutputDims.emplace_back(network->getOutput(0)->getDimensions());
    mOutputDims.emplace_back(network->getOutput(1)->getDimensions());

    return true;
}

//!
//! \brief Creates a network with a single custom layer containing the NonZero plugin and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the NonZero plugin
//!
//! \param builder Pointer to the engine builder
//!
bool TestFusedAttnSpPredictionPlugin::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    // 1. Specify the input tensor
    int32_t const batchsize = 1;
    int32_t const n_queries = 300;
    int32_t const n_channels = 256;
    auto* in = network->addInput(
        "Input", 
        /*
        NOTE: We manually mark the input type as float, whereas the input is actually int8.
        The reason is that if we mark input as int8, TensorRT recongonizes the layer is quantized, and
        it requires Q/DQ nodes, calibrator or dynamic range setting.
        */
        // DataType::kINT8,
        DataType::kFLOAT,
        {3, {batchsize, n_queries, n_channels}}
    );
    /*
    Set dynamic range (min, max) for INT8 input, or:
        [E] [TRT] IBuilder::buildSerializedNetwork: Error Code 4: API Usage Error (Input: input/output with DataType Int8 in network without Q/DQ layers must have dynamic range set when no calibrator is used.)
    */ 
    // in->setDynamicRange(-128.0f, 127.0f);  // Typical INT8 range
    ASSERT(in != nullptr);

    // 2. Create the plugin through creator
   
    // config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 >> 30);
    // NOTE: Huge mistake made here. "&mParams.input_scale" is not correct.
    std::vector<PluginField> const vecPF{
        // {"rowOrder", &mParams.rowOrder, PluginFieldType::kINT32, 1}
        {"input_scale", mParams.input_scale.data(), PluginFieldType::kFLOAT32, 1},
        {"input_offset", mParams.input_offset.data(), PluginFieldType::kINT8, 1},
        {"sp_weight_scale", mParams.sp_weight_scale.data(), PluginFieldType::kFLOAT32, 192},
        {"sp_weight_offset", mParams.sp_weight_offset.data(), PluginFieldType::kINT8, 192},
        {"attn_weight_scale", mParams.attn_weight_scale.data(), PluginFieldType::kFLOAT32, 96},
        {"attn_weight_offset", mParams.attn_weight_offset.data(), PluginFieldType::kINT8, 96},
        {"int8_sp_weight", mParams.int8_sp_weight.data(), PluginFieldType::kINT8, 192 * 256},
        {"int8_attn_weight", mParams.int8_attn_weight.data(), PluginFieldType::kINT8, 96 * 256},
        {"sp_fp32_bias", mParams.sp_fp32_bias.data(), PluginFieldType::kFLOAT32, 192},
        {"attn_fp32_bias", mParams.attn_fp32_bias.data(), PluginFieldType::kFLOAT32, 96}
    };
    PluginFieldCollection pfc{static_cast<int32_t>(vecPF.size()), vecPF.data()};

    // 2.1 Get the plugin creator TODO: Where do we pass the creator to registry?
    auto pluginCreator = static_cast<IPluginCreatorV3One*>(
        getPluginRegistry()->getCreator(
            PLUGIN_NAME, "1", PLUGIN_NAMESPACE
        )
    );
    auto plugin = std::unique_ptr<IPluginV3>(
        pluginCreator->createPlugin(PLUGIN_NAME, &pfc, TensorRTPhase::kBUILD)
    );

    std::vector<ITensor*> inputsVec{in};
    auto pluginNonZeroLayer = network->addPluginV3(
        inputsVec.data(), inputsVec.size(), nullptr, 0, *plugin
    );
    ASSERT(pluginNonZeroLayer != nullptr);
    ASSERT(pluginNonZeroLayer->getOutput(0) != nullptr);
    ASSERT(pluginNonZeroLayer->getOutput(1) != nullptr);
    pluginNonZeroLayer->setName("FusedAttnSpPredictionPluginLayer");

    pluginNonZeroLayer->getOutput(0)->setName("Output0");
    pluginNonZeroLayer->getOutput(1)->setName("Output1");

    network->markOutput(*(pluginNonZeroLayer->getOutput(0)));
    network->markOutput(*(pluginNonZeroLayer->getOutput(1)));

    pluginNonZeroLayer->getOutput(0)->setType(DataType::kFLOAT);
    pluginNonZeroLayer->getOutput(1)->setType(DataType::kFLOAT);
    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool TestFusedAttnSpPredictionPlugin::infer()
{

    // NOTE: Must allocate sufficient buffer to hold input and output
    // Input: (bs, 300, 256)
    // Output: (bs, 300, 96) and (bs, 300, 192)
    std::vector<int64_t> ioVolumes = 
    {
            volume(mInputDims[0]),
            volume(mOutputDims[0]),
            volume(mOutputDims[1])
    };

    // TODO: Understand RAII here
    samplesCommon::BufferManager buffers(mEngine, ioVolumes);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    bool status = context->enqueueV3(stream);
    if (!status)
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers.
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete.
    CHECK(cudaStreamSynchronize(stream));

    // Release stream.
    CHECK(cudaStreamDestroy(stream));

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool TestFusedAttnSpPredictionPlugin::processInput(samplesCommon::BufferManager const& buffers)
{
    int64_t const input_size = volume(mInputDims[0]);
    bool use_true_input_data = true;
    if (use_true_input_data)
    {
        std::string input_path = locateFile("input_int8.bin", mParams.dataDirs);
        if (input_path.empty())
        {
            sample::gLogError << "Could not find file: input_int8.bin" << std::endl;
            return false;
        }
        std::ifstream file(input_path, std::ios::binary);
        if (!file)
        {
            sample::gLogError << "Could not open file: input_int8.bin" << std::endl;
            return false;
        }
        // Read the input data from the file
        std::vector<int8_t> inputData(input_size);
        file.read(reinterpret_cast<char*>(inputData.data()),  input_size* sizeof(int8_t));

        char* hostDataBuffer = static_cast<char*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
        for (int32_t i = 0; i < input_size; ++i)
        {

            hostDataBuffer[i] = inputData[i];
        }
        sample::gLogInfo << "First 10 Input:" << std::endl;
        for(int32_t i = 0; i < 10; ++i)
        {
            sample::gLogInfo << (int32_t)(hostDataBuffer[i]) << " ";
        }
        sample::gLogInfo << std::endl;
    }
    // use pseudo input just for debugging
    else 
    {
        std::vector<float> inputData(input_size);
        for (auto &v: inputData)
        {
            v = static_cast<float>(1.3);
        }
        
        float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
        for (int32_t i = 0; i < input_size; ++i)
        {
            hostDataBuffer[i] = inputData[i];
        }
        sample::gLogInfo << "First 10 Input:" << std::endl;
        for(int32_t i = 0; i < 10; ++i)
        {
            sample::gLogInfo << (hostDataBuffer[i]) << " ";
        }
        sample::gLogInfo << std::endl;
    }


    return true;
}

//!
//! \brief Verify result
//!
//! \return whether the output correctly identifies all (and only) non-zero elements
//!
bool TestFusedAttnSpPredictionPlugin::verifyOutput(samplesCommon::BufferManager const& buffers)
{
    std::unordered_map<std::string, std::pair<int32_t, std::string>> outputs_path_map {
        {"Output0", {0, "output_attn.bin"}},
        {"Output1", {1, "output_sp.bin"}}
    };
    for (auto const& [name, index_path]: outputs_path_map)
    {
        int32_t output_idx = index_path.first;
        std::string path = index_path.second;
        std::string output_path = locateFile(path, mParams.dataDirs);
        if (output_path.empty())
        {
            sample::gLogError << "Could not find file: " << path << std::endl;
            return false;
        }
        std::ifstream file(output_path, std::ios::binary);
        if (!file)
        {
            sample::gLogError << "Could not open file: " << path << std::endl;
            return false;
        }
        // Read the output data from the file
        int64_t const output_size = volume(mOutputDims[output_idx]);
        std::vector<float> outputDataGT(output_size);
        file.read(reinterpret_cast<char*>(outputDataGT.data()),  output_size* sizeof(float));
        file.close();
        
        // Read the output data from the buffer
        float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(name));
        std::vector<float> output(hostDataBuffer, hostDataBuffer + output_size);
        
        // compare the output data with the ground truth
        float max_diff = 0.1f;
        for (int32_t i = 0; i < output_size; ++i)
        {
            float diff = std::abs(output[i] - outputDataGT[i]);
            if (diff > max_diff)
            {
                sample::gLogError << name << ": " <<  "Output data does not match ground truth. Max diff: " << diff << std::endl;
                return false;
            }
        }

    }
    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
FusedAttnSpParams initializeSampleParams(samplesCommon::Args const& args)
{
    FusedAttnSpParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        std::cerr << "No data directories provided. Please specify the parameters weight path" << std::endl;
        std::abort();
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.inputTensorNames.push_back("Input");
    params.outputTensorNames.push_back("Output0");
    params.outputTensorNames.push_back("Output1");
    params.fp16 = args.runInFp16;
    ASSERT(!params.fp16); // Do not use fp16 for now
    // Load parameters from local file
    std::unordered_map<std::string, std::string> pathMap{
        {"input_scale", "input_scale.bin"},
        {"input_offset", "input_offset.bin"},
        {"sp_weight_scale", "sp_weight_scale.bin"},
        {"sp_weight_offset", "sp_weight_offset.bin"},
        {"attn_weight_scale", "attn_weight_scale.bin"},
        {"attn_weight_offset", "attn_weight_offset.bin"},
        {"int8_sp_weight", "int8_sp_weight.bin"},
        {"int8_attn_weight", "int8_attn_weight.bin"},
        {"sp_fp32_bias", "sp_fp32_bias.bin"},
        {"attn_fp32_bias", "attn_fp32_bias.bin"}
    };
    for (auto const& [key, value] : pathMap)
    {
        std::string path = locateFile(value, params.dataDirs);
        if (path.empty())
        {
            sample::gLogError << "Could not find file: " << value << std::endl;
            return params;
        }
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            sample::gLogError << "Could not open file: " << value << std::endl;
            return params;
        }
        if (key == "input_scale")
        {
            params.input_scale.resize(1);
            file.read(reinterpret_cast<char*>(params.input_scale.data()), sizeof(float));
        }
        else if (key == "input_offset")
        {
            params.input_offset.resize(1);
            file.read(reinterpret_cast<char*>(params.input_offset.data()), sizeof(int8_t));
        }
        else if (key == "sp_weight_scale")
        {
            params.sp_weight_scale.resize(192);
            file.read(reinterpret_cast<char*>(params.sp_weight_scale.data()), 192 * sizeof(float));
        }
        else if (key == "sp_weight_offset")
        {
            params.sp_weight_offset.resize(192);
            file.read(reinterpret_cast<char*>(params.sp_weight_offset.data()), 192 * sizeof(int8_t));
        }
        else if (key == "attn_weight_scale")
        {
            params.attn_weight_scale.resize(96);
            file.read(reinterpret_cast<char*>(params.attn_weight_scale.data()), 96 * sizeof(float));
        }
        else if (key == "attn_weight_offset")
        {
            params.attn_weight_offset.resize(96);
            file.read(reinterpret_cast<char*>(params.attn_weight_offset.data()), 96 * sizeof(int8_t));
        }
        else if (key == "int8_sp_weight")
        {
            params.int8_sp_weight.resize(192 * 256);
            file.read(reinterpret_cast<char*>(params.int8_sp_weight.data()), 192 * 256 * sizeof(int8_t));
        }
        else if (key == "int8_attn_weight")
        {
            params.int8_attn_weight.resize(96 * 256);
            file.read(reinterpret_cast<char*>(params.int8_attn_weight.data()), 96 * 256 * sizeof(int8_t));
        }
        else if (key == "sp_fp32_bias")
        {
            params.sp_fp32_bias.resize(192);
            file.read(reinterpret_cast<char*>(params.sp_fp32_bias.data()), 192 * sizeof(float));
        }
        else if (key == "attn_fp32_bias")
        {
            params.attn_fp32_bias.resize(96);
            file.read(reinterpret_cast<char*>(params.attn_fp32_bias.data()), 96 * sizeof(float));
        }
        file.close();
    }

    return params;

}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_non_zero_plugin [-h or --help] [-d or --datadir=<path to data directory>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
    std::cout << "--columnOrder   Run plugin in column major output mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(kSAMPLE_NAME, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    TestFusedAttnSpPredictionPlugin sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Ink plugin" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    printf("\e[32m[Congrats]\n");
    return sample::gLogger.reportPass(sampleTest);
}
