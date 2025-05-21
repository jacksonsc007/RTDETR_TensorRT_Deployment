/*
This implementation loads plugins for runtime. It proves that the segmentation fault actually results from this
missing step.

Here is the link of the official doc:
https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html#plugin-shared-libraries:~:text=runtime%2D%3EgetPluginRegistry().loadLibrary(pluginLib.c_str())
*/
// #define DEFINE_TRT_ENTRYPOINTS 1

#include <cstdlib>
#include <iostream>
#include <fstream>

#include <NvOnnxParser.h>

#include "rtdetr.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include "utils.h"

// sample helper functions
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "half.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "sampleOptions.h"

using namespace nvinfer1;



RTDetr::RTDetr(
        const std::string trtFile,
        const std::string onnxFile,
        int gpuId,
        float nmsThresh,
        float confThresh,
        int numClass
    ): trtFile_(trtFile), onnxFile_(onnxFile), nmsThresh_(nmsThresh), confThresh_(confThresh), numClass_(numClass)
{

#ifdef DEFINE_TRT_ENTRYPOINTS
    printf("#define TRT_ENTRYPOINTS\n");
#endif

    printf("\e[31m[INFO]\e[m Creating RTDetr object.\n");
    gLogger = Logger(ILogger::Severity::kINFO);
    cudaSetDevice(gpuId);
    CHECK(cudaStreamCreate(&stream));



    // load engine
    if (get_engine())
    {
        printf("\e[31m[ERROR]\e[m Failed to load engine.\n");
        std::abort();
    }

    try
    {
        if (engine == nullptr)
        {
            std::cerr << "\e[31m[ERROR]\e[m Engine is null." << std::endl;
            std::abort();
        }

        context = engine->createExecutionContext();
    }
    catch (const std::exception& e)
    {
        std::cerr << "\e[31m[ERROR]\e[m Exception while creating execution context: " << e.what() << std::endl;
        std::abort();
    }
    catch (...)
    {
        std::cerr << "\e[31m[ERROR]\e[m Unknown exception while creating execution context." << std::endl;
        std::abort();
    }
    int32_t nb_bindings = engine->getNbIOTensors();
    // const char * binding_name;

    // Show input and output info
    // for (int32_t i = 0; i < nb_bindings; ++i)
    // {
    //     // get binding names, dims, type
    //     const char * binding_name = engine->getIOTensorName(i);
    //     Dims shape = engine->getTensorShape(binding_name);
    //     printf("Binding %d Name: %s, NbDims: %d, Type: %d\n", 
    //         i, binding_name, shape.nbDims, engine->getTensorDataType(binding_name)
    //     );
    // }

    printf("\n\e[31m[INFO]\e[m Input and Output information:\n");
    auto num = engine->getNbIOTensors();
    for (int32_t i = 0; i < num; i++){
        std::cout << std::left << std::setw(20) << engine->getIOTensorName(i);
        auto shape = engine->getTensorShape(engine->getIOTensorName(i));
        std::cout << "Tensor shape: [";
        
        // Create a string to hold the shape dimensions
        std::string shapeStr;
        for (auto j = 0; j < shape.nbDims; j++){
            shapeStr += std::to_string(shape.d[j]);
            if (j < shape.nbDims - 1) shapeStr += ", ";
        }
        
        // Pad the shape string for alignment
        std::cout << std::left << std::setw(20) << (shapeStr + "]");
        
        // Print the format with alignment
        std::cout << "Format: " << engine->getTensorFormatDesc(engine->getIOTensorName(i)) << std::endl;
    }
    printf("\n");

    // NOTE: Define our expected input/output specification. Significant!!!
    std::vector<TypeSpec> ExpectedFormat = {
        // images
        TypeSpec{DataType::kFLOAT, TensorFormat::kLINEAR, "KLINEAR"},
        // orig_target_sizes
        TypeSpec{DataType::kINT64, TensorFormat::kLINEAR, "KLINEAR"},
        // labels
        TypeSpec{DataType::kINT64, TensorFormat::kLINEAR, "KLINEAR"},
        // scores
        TypeSpec{DataType::kFLOAT, TensorFormat::kLINEAR, "KLINEAR"},
        // boxes
        TypeSpec{DataType::kFLOAT, TensorFormat::kLINEAR, "KLINEAR"},
    };

    if (!verify(ExpectedFormat))
    {
        printf("\e[31m[ERROR]\e[m Data type mismatch\n");
        exit(1); // Terminate the program with an error code
    }

    // There are 2 inputs and 3 outputs for rt-detr, which could be observed from onnx visualization
    // context->setBindingDimensions(0, Dims64 {4, {1, 3, kInputH, kInputW}});
    // context->setBindingDimensions(1, Dims64 {2, {1, 2}});
    
    // We don't need to set the dimension of output, which induces segment fault.
    // Input shape is determined by the ihput.
    // context->setBindingDimensions(2, Dims32 {2, {1, 300}});
    // context->setBindingDimensions(3, Dims32 {3, {1, 300, 4}});
    // context->setBindingDimensions(4, Dims32 {2, {1, 300}});
    

    
    // get engine output info
    OUTPUT_CANDIDATES = 300;
    assert (OUTPUT_CANDIDATES == 300);

    // int outputSize = 1;  // 84 * 8400
    // for (int i = 0; i < outDims.nbDims; i++){
    //     outputSize *= outDims.d[i];
    // }

    // prepare output data space on host
    // outputData = new float[1 + kMaxNumOutputBbox * kNumBoxElement];
    // 
    boxes_h = new float[OUTPUT_CANDIDATES * 4];
    labels_h = new int64_t[OUTPUT_CANDIDATES];
    scores_h = new float[OUTPUT_CANDIDATES];

    // prepare input and output space on device
    vBufferD.resize(5, nullptr);
    CHECK(cudaMalloc(&vBufferD[0], 3 * kInputH * kInputW * sizeof(float))); // images (1, 3, 640, 640)
    CHECK(cudaMalloc(&vBufferD[1], 1 * 2 * sizeof(int64_t))); // orig_target_sizes (1,2)
    CHECK(cudaMalloc(&vBufferD[2], OUTPUT_CANDIDATES * sizeof(float))); // scores (1, 300)
    CHECK(cudaMalloc(&vBufferD[3], OUTPUT_CANDIDATES * sizeof(int64_t))); // labels (1, 300)
    CHECK(cudaMalloc(&vBufferD[4], 4 * OUTPUT_CANDIDATES * sizeof(float))); // boxes (1, 300, 4)

}

RTDetr::RTDetr(RTDetr & other){
    printf("\e[31m[INFO]\e[m Copy RTDetr object.\n");
}

int32_t RTDetr::get_engine(){
    /*
    Ink: If there is a available engine, we directly use it, or we build from onnx file.
    */

    // Load custom plugin lib
    printf("\e[31m[INFO]\e[m Loading Custom shared Lib\n");

    std::string pluginLibs = "shared_plugin_libs/libfused_attn_offset_prediction_plugin_v2.so";
    // use this if custom plugin is under custom namespace
    // using LibraryPtr = std::unique_ptr<samplesCommon::DynamicLibrary>;
    // // NOTE: `static` must not be omitted, otherwise the library will be unloaded after the function exits, causing segementation fault (along with no other useful error hint).
    // LibraryPtr nvinferPluginLib{};
    // nvinferPluginLib = samplesCommon::loadLibrary(pluginLibs.c_str());
    // auto pInitLibNvinferPlugins
    //     = nvinferPluginLib->symbolAddress<bool(void*, char const*)>("initLibNvInferPlugins");
    // ASSERT(pInitLibNvinferPlugins != nullptr);
    //     pInitLibNvinferPlugins(&sample::gLogger.getTRTLogger(), "ink_plugins");
    if (access(trtFile_.c_str(), F_OK) == 0)
    {
        printf("\e[31m[INFO] \e[mSucceeded in in finding engine file. Trying to load ... \n");

        std::ifstream engineFile(trtFile_, std::ios::binary);
        long int fsize = 0;


        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize); // extract from stream and store to engineString
        if (engineString.size() == 0) { printf("\e[31m[ERROR]\e[m Failed getting serialized engine!\n"); return -1; }
        printf("\e[32m[INFO]\e[m Engine file loaded successfully.\n");

        runtime = createInferRuntime(gLogger);
        // NOTE: load plugins for runtime. This is crucial, or sementation fault will occur.
        runtime->getPluginRegistry().loadLibrary(pluginLibs.c_str());
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) 
        {
            printf("\e[31m[ERROR]\e[m Failed building engine!\n"); return -1; 
        }
        printf("\e[32m[INFO]\e[m The engine is deserailized successfully.\n");
        return 0;
    } 
    else 
    {
        printf("\e[31m[INFO]\e[m Failed to find engine file. Build from onnx file %s...\n", onnxFile_.c_str());
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile* profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30);
        
        // NOTE: serailize plugins with engine
        const char* pluginLibsArr[] = { pluginLibs.c_str() };
        // config->setPluginsToSerialize(pluginLibsArr, 1);

        IInt8Calibrator *     pCalibrator = nullptr;
        if (bFP16Mode){
            printf("\e[31m[INFO]\e[m FP16 mode enabled\n");
            config->setFlag(BuilderFlag::kFP16);
        }
        else if (bINT8Mode){
            printf("\e[31m[INFO]\e[m INT8 mode enabled\n");
            config->setFlag(BuilderFlag::kINT8);
            int batchSize = 8;
            pCalibrator = new Int8EntropyCalibrator2(batchSize, kInputW, kInputH, calibrationDataPath.c_str(), cacheFile.c_str());
            config->setInt8Calibrator(pCalibrator);
        }
        else {
            printf("\e[31m[INFO]\e[m fp32 mode enabled\n");
        }

        // Load custom plugin lib under default namespace
        // getPluginRegistry()->loadLibrary(pluginLibs.c_str());
        builder->getPluginRegistry().loadLibrary(pluginLibs.c_str());
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile_.c_str(), int(gLogger.reportableSeverity))){
            printf("\e[31m[ERROR]\e[m Failed parsing .onnx file!\n");
            for (int i = 0; i < parser->getNbErrors(); ++i){
                auto *error = parser->getError(i);
                printf("\e[31m[ERROR]\e[m %s\n", std::to_string(int(error->code())).c_str(), error->desc());
            }
            return -1;
        }
        printf("\e[32m[INFO]\e[m Succeeded in parsing .onnx file!\n");

        printf("\e[31m[INFO]\e[m Input and Output information:\n");
        int32_t nb_input = network->getNbInputs();
        ITensor* inputTensor, *outputTensor;
        Dims inputDims, outputDims;
        int32_t nbDims;
        for (int32_t i = 0; i < nb_input; i++)
        {
            inputTensor = network->getInput(i);
            inputDims  = inputTensor->getDimensions();
            nbDims = inputDims.nbDims;
            printf("Input  Tensor Name: %20s with %2d dims and type %2d: [", inputTensor->getName(), nbDims, inputTensor->getType());
            for (int32_t j = 0; j < nbDims; j++)
            {
                printf(" %ld ", inputDims.d[j]);
            }
            printf("]\n");
        }

        int32_t nb_output = network->getNbOutputs();
        for (int32_t i = 0; i < nb_output; i++)
        {
            outputTensor = network->getOutput(i);
            outputDims  = outputTensor->getDimensions();
            nbDims = outputDims.nbDims;
            printf("Output Tensor Name: %20s with %2d dims and type %2d: [", outputTensor->getName(), nbDims, outputTensor->getType());
            for (int32_t j = 0; j < nbDims; j++)
            {
                printf(" %ld ", outputDims.d[j]);
            }
            printf("]\n");
        }
        
        
        profile->setDimensions("images", OptProfileSelector::kMIN, Dims {4, {1, 3, kInputH, kInputW}}); 
        profile->setDimensions("images", OptProfileSelector::kOPT, Dims {4, {1, 3, kInputH, kInputW}});
        profile->setDimensions("images", OptProfileSelector::kMAX, Dims {4, {1, 3, kInputH, kInputW}});
        profile->setDimensions("orig_target_sizes", OptProfileSelector::kMIN, Dims {2, {1, 2}}); 
        profile->setDimensions("orig_target_sizes", OptProfileSelector::kOPT, Dims {2, {1, 2}});
        profile->setDimensions("orig_target_sizes", OptProfileSelector::kMAX, Dims {2, {1, 2}});
        
        config->addOptimizationProfile(profile);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        printf("\e[32m[INFO]\e[m Succeeded in in serializing engine!\n");

        runtime = createInferRuntime(gLogger);
        // NOTE: load plugins for runtime. This is crucial, or sementation fault will occur.
        runtime->getPluginRegistry().loadLibrary(pluginLibs.c_str());
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) 
        {
            printf("\e[31m[ERROR]\e[m Failed building engine!\n");
            return -1; 
        }
        printf("\e[32m[INFO]\e[m Succeeded in building engine!\n");
        if (bINT8Mode && pCalibrator != nullptr){
            delete pCalibrator;
        }

        std::ofstream engineFile(trtFile_, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        printf("\e[31m[INFO]\e[m Succeeded in saving built engine %s\n", trtFile_.c_str());

        delete engineString;
        delete parser;
        delete config;
        delete network;
        delete builder;
        return 0;
    }
}

RTDetr::~RTDetr(){
    cudaStreamDestroy(stream);

    for (int i = 0; i < 5; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    // CHECK(cudaFree(transposeDevice));
    // CHECK(cudaFree(decodeDevice));

    // delete [] outputData;
    
    delete [] boxes_h;
    delete [] scores_h;
    delete [] labels_h;

    delete context;
    // TODO: if delete, errors occured
    delete engine;
    delete runtime;
    printf("\e[31m[INFO]\e[m RT-DETR object destroyed.\n");
}

std::vector<Detection> RTDetr::inference(cv::Mat& img){
    if (img.empty()) return {};

    // Get the original image size
    int srcHeight = img.rows;
    int srcWidth = img.cols;

    // context = engine->createExecutionContext();
    

    // CHECK(cudaMalloc(&transposeDevice, outputSize * sizeof(float)));
    // CHECK(cudaMalloc(&decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float)));
    


    // Initialize the input
    // NOTE: the normalization coeffs need to be validated.
    // put input on device, then letterbox、bgr to rgb、hwc to chw、normalize.
    preprocess(img, (float*)vBufferD[0], kInputH, kInputW, stream);
    // int64_t orig_target_sizes_h[2] = {srcWidth, srcHeight};
    int64_t orig_target_sizes_h[2] = {kInputW, kInputH};
    CHECK(cudaMemcpyAsync(vBufferD[1], orig_target_sizes_h, 2 * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    bool status;
    status = context->setTensorAddress("images", vBufferD[0]);
    assert (status == true);
    status = context->setTensorAddress("orig_target_sizes", vBufferD[1]);
    assert (status == true);
    status = context->setTensorAddress("scores", vBufferD[2]);
    assert (status == true);
    status = context->setTensorAddress("labels", vBufferD[3]);
    assert (status == true);
    status = context->setTensorAddress("boxes", vBufferD[4]);
    assert (status == true);
    status = context->enqueueV3(stream);
    assert (status == true);
    // tensorrt inference
    context->enqueueV3(stream);
    // context->enqueueV2(vBufferD.data(), stream, nullptr);

    // transpose [1 84 8400] convert to [1 8400 84]
    // transpose((float*)vBufferD[1], transposeDevice, OUTPUT_CANDIDATES, numClass_ + 4, stream);

    // convert [1 8400 84] to [1 7001]
    // decode(transposeDevice, decodeDevice, OUTPUT_CANDIDATES, numClass_, confThresh_, kMaxNumOutputBbox, kNumBoxElement, stream);
    // cuda nms
    // nms(decodeDevice, nmsThresh_, kMaxNumOutputBbox, kNumBoxElement, stream);



    // CHECK(cudaMemcpyAsync(outputData, decodeDevice, (1 + kMaxNumOutputBbox * kNumBoxElement) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // 
    CHECK(cudaMemcpyAsync(scores_h, vBufferD[2], OUTPUT_CANDIDATES * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(labels_h, vBufferD[3], OUTPUT_CANDIDATES * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(boxes_h, vBufferD[4], OUTPUT_CANDIDATES * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    

    cudaStreamSynchronize(stream);

    std::vector<Detection> vDetections;
    // int count = std::min((int)outputData[0], kMaxNumOutputBbox);
    int count = OUTPUT_CANDIDATES;
    float score;
    int32_t label;
    for (int i = 0; i < count; i++)
    {
        score = scores_h[i];
        label = labels_h[i];
        if (score > confThresh_)
        {
            Detection det;
            memcpy(det.bbox, boxes_h + i * 4, 4 * sizeof(float));
            det.conf = score;
            det.classId = label;
            vDetections.push_back(det);
        }
    }

    // for (int i = 0; i < count; i++){
    //     int pos = 1 + i * kNumBoxElement;
    //     int keepFlag = (int)outputData[pos + 6];
    //     if (keepFlag == 1){
    //         Detection det;
    //         memcpy(det.bbox, &outputData[pos], 4 * sizeof(float));
    //         det.conf = outputData[pos + 4];
    //         det.classId = (int)outputData[pos + 5];
    //         vDetections.push_back(det);
    //     }
    // }

    for (size_t j = 0; j < vDetections.size(); j++){
        scale_bbox(img, vDetections[j].bbox);
    }

    return vDetections;
}

void RTDetr::draw_image(cv::Mat& img, std::vector<Detection>& inferResult){
    // draw inference result on image
    for (size_t i = 0; i < inferResult.size(); i++){
        cv::Scalar bboxColor(get_random_int(), get_random_int(), get_random_int());
        cv::Rect r(
            round(inferResult[i].bbox[0]),
            round(inferResult[i].bbox[1]),
            round(inferResult[i].bbox[2] - inferResult[i].bbox[0]),
            round(inferResult[i].bbox[3] - inferResult[i].bbox[1])
        );
        cv::rectangle(img, r, bboxColor, 2);

        std::string className = vClassNames[(int)inferResult[i].classId];
        std::string labelStr = className + " " + std::to_string(inferResult[i].conf).substr(0, 4);

        cv::Size textSize = cv::getTextSize(labelStr, cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
        cv::Point topLeft(r.x, r.y - textSize.height - 3);
        cv::Point bottomRight(r.x + textSize.width, r.y);
        cv::rectangle(img, topLeft, bottomRight, bboxColor, -1);
        cv::putText(img, labelStr, cv::Point(r.x, r.y - 2), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
}


bool RTDetr::verify(std::vector<TypeSpec> const& specCollection)
{
    assert(engine->getNbIOTensors() == 5);
    std::vector<char const *> IONames = {
        "images",
        "orig_target_sizes",
        "scores",
        "labels",
        "boxes"
    };

    // helper function to verify data types
    auto verifyType = [](char const * name, DataType actual, DataType expected) {
        if (actual != expected)
        {
            sample::gLogError << name << " expected " << expected << " data type, got " << actual;
            return false;
        }
        return true;
    };
    // helper function to verify data format
    auto verifyFormat = [](char const * name, std::string actual, std::string expected) {
        if (expected.find(actual) != std::string::npos)
        {
            sample::gLogError << name<< " expected " << expected << " format, got " << actual;
            return false;
        }
        return true;
    };

    for (int32_t i = 0; i < engine->getNbIOTensors(); i++)
    {
        char const* IOName = engine->getIOTensorName(i);
        if (!verifyType(IOName, engine->getTensorDataType(IOName), specCollection[i].dtype))
        {
            return false;
        }
        if (!verifyFormat(IOName, engine->getTensorFormatDesc(IOName), specCollection[i].formatName))
        {
            return false;
        }
        
    }


    return true;
}