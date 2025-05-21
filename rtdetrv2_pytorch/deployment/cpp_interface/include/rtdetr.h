#ifndef RTDETR_INFER_H
#define RTDETR_INFER_H

#include <opencv2/opencv.hpp>
#include "public.h"
#include "rtdetr_config.h"
#include "types.h"

using namespace nvinfer1;

//! Specification for a network I/O tensor.
class TypeSpec
{
public:
    DataType dtype;         //!< datatype
    TensorFormat format;    //!< format
    std::string formatName; //!< name of the format
};


class RTDetr
{
public:
    RTDetr(
        const std::string trtFile,
        const std::string onnxFile,
        int gpuId=kGpuId,
        float nmsThresh=kNmsThresh,
        float confThresh=kConfThresh,
        int numClass=kNumClass
    );
    RTDetr(RTDetr& other);
    ~RTDetr();
    std::vector<Detection> inference(cv::Mat& img);
    static void draw_image(cv::Mat& img, std::vector<Detection>& inferResult);

    bool verify(std::vector<TypeSpec> const& specCollection);

private:
    int32_t get_engine();

private:
    Logger              gLogger;
    std::string         trtFile_;
    std::string         onnxFile_;

    int                 numClass_;
    float               nmsThresh_;
    float               confThresh_;

    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    cudaStream_t        stream;

    // float *             outputData;
    std::vector<void *> vBufferD;
    // float *             transposeDevice;
    // float *             decodeDevice;
    
    float *             boxes_h;
    float *             scores_h;
    int64_t *           labels_h;

    int                 OUTPUT_CANDIDATES;  // 8400: 80 * 80 + 40 * 40 + 20 * 20
    // std::string pluginLibs = "shared_plugin_libs/tensorrt_10_9/libfused_attn_offset_prediction_plugin_v3.so";
    std::string pluginLibs = "shared_plugin_libs/tensorrt_10_7/libfused_attn_offset_prediction_plugin_v3.so";
    // std::string pluginLibs = "shared_plugin_libs/tensorrt_10_7/libfused_attn_offset_prediction_plugin_v2.so";
    // std::string pluginLibs = "shared_plugin_libs/tensorrt_10_7/libfused_attn_offset_prediction_plugin_v3.so";

};

#endif  // INFER_H
