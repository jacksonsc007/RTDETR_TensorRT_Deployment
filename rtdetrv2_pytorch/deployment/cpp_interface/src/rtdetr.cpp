/*
This implementation is based on implementation 3. It demonstrate that
```cpp
    builder->getPluginRegistry().loadLibrary(pluginLibs.c_str());
    runtime->getPluginRegistry().loadLibrary(pluginLibs.c_str());
```
suffice to load plugins with default and custom namespace, as long as the following inferance is implemented:
```cpp
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

```



Consequently, the following loading snippet is no longer needed:
```cpp

    // use this if custom plugin is under custom namespace
    using LibraryPtr = std::unique_ptr<samplesCommon::DynamicLibrary>;
    // NOTE: `static` must not be omitted, otherwise the library will be unloaded after the function exits, causing segementation fault (along with no other useful error hint).
    LibraryPtr nvinferPluginLib{};
    nvinferPluginLib = samplesCommon::loadLibrary(pluginLibs.c_str());
    auto pInitLibNvinferPlugins
        = nvinferPluginLib->symbolAddress<bool(void*, char const*)>("initLibNvInferPlugins");
    ASSERT(pInitLibNvinferPlugins != nullptr);
        pInitLibNvinferPlugins(&sample::gLogger.getTRTLogger(), "ink_plugins");
        
```
*/
// #define DEFINE_TRT_ENTRYPOINTS 1

#include <cstddef>
#include <cstdint>
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

// profiling
#include "profile.h"
#include <filesystem>
using namespace nvinfer1;


static int64_t volume(Dims const& dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

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
    numIO = engine->getNbIOTensors();
    std::vector<int64_t> volumnIO(numIO, 0);
    std::vector<int32_t> datasizeIO(numIO, 0);
    IOnames.resize(numIO);
    for (int32_t i = 0; i < numIO; i++){
        std::string name = engine->getIOTensorName(i);
        IOnames[i] = name;
        
        std::cout << std::left << std::setw(20) << name;
        auto shape = engine->getTensorShape(name.c_str());
        volumnIO[i] = volume(shape);
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
        std::cout << "Format: " << engine->getTensorFormatDesc(name.c_str()) << std::endl;
        auto datatype = engine->getTensorDataType(name.c_str());
        if (datatype == nvinfer1::DataType::kINT8) datasizeIO[i] = 8;
        else if (datatype == nvinfer1::DataType::kFLOAT) datasizeIO[i] = 32;
        else if (datatype == nvinfer1::DataType::kINT64) datasizeIO[i] = 64;
        else
        {
            printf("Unsupported datatype: %d, please modify codes\n", datatype);
            std::abort();
        }

    }
    printf("\n");

    // NOTE: Define our expected input/output specification. Significant!!!
    // std::vector<TypeSpec> ExpectedFormat = {
    //     // images
    //     TypeSpec{DataType::kFLOAT, TensorFormat::kLINEAR, "KLINEAR"},
    //     // orig_target_sizes
    //     TypeSpec{DataType::kINT64, TensorFormat::kLINEAR, "KLINEAR"},
    //     // labels
    //     TypeSpec{DataType::kINT64, TensorFormat::kLINEAR, "KLINEAR"},
    //     // scores
    //     TypeSpec{DataType::kFLOAT, TensorFormat::kLINEAR, "KLINEAR"},
    //     // boxes
    //     TypeSpec{DataType::kFLOAT, TensorFormat::kLINEAR, "KLINEAR"},
    // };

    // if (!verify(ExpectedFormat))
    // {
    //     printf("\e[31m[ERROR]\e[m Data type mismatch\n");
    //     exit(1); // Terminate the program with an error code
    // }

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
    vBufferD.resize(numIO, nullptr);
    for (int i = 0; i < numIO; ++i)
    {
        int64_t volume = volumnIO[i];
        int32_t datasize = datasizeIO[i];
        CHECK(
            cudaMalloc(
                &vBufferD[i],
                volume * datasize
            )
        );
    }
    // CHECK(cudaMalloc(&vBufferD[0], 3 * kInputH * kInputW * sizeof(float))); // images (1, 3, 640, 640)
    // CHECK(cudaMalloc(&vBufferD[1], 1 * 2 * sizeof(int64_t))); // orig_target_sizes (1,2)
    // CHECK(cudaMalloc(&vBufferD[2], OUTPUT_CANDIDATES * sizeof(float))); // scores (1, 300)
    // CHECK(cudaMalloc(&vBufferD[3], OUTPUT_CANDIDATES * sizeof(int64_t))); // labels (1, 300)
    // CHECK(cudaMalloc(&vBufferD[4], 4 * OUTPUT_CANDIDATES * sizeof(float))); // boxes (1, 300, 4)

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
        // The default library is based on TensorRT 10.9 and IPluginv3.
        if (runtime->getPluginRegistry().loadLibrary(pluginLibs.c_str()) == nullptr)
        {
            printf("\e[31m[ERROR]\e[m Failed loading plugin library!\n");
            return -1;
        };
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
        // const char* pluginLibsArr[] = { pluginLibs.c_str() };
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

        // Load custom plugin lib under both default namespace and custom namespace
        if (builder->getPluginRegistry().loadLibrary(pluginLibs.c_str()) == nullptr)
        {
            printf("\e[31m[ERROR]\e[m Failed loading plugin library!\n");
            return -1;
        };
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
        if (runtime->getPluginRegistry().loadLibrary(pluginLibs.c_str()) == nullptr)
        {
            printf("\e[31m[ERROR]\e[m Failed loading plugin library!\n");
            return -1;
        };
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

        std::string outputDir = "built_engine/";
        if (!std::filesystem::exists(outputDir))
        {
            std::filesystem::create_directory(outputDir); 
        }
        trtFile_ = "default_built.engine";
        std::ofstream engineFile(outputDir + trtFile_, std::ios::binary);
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

    for (int i = 0; i < vBufferD.size(); ++i)
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

    /* Initialize the input
    Put input from cpu memory to GPU memory, then apply letterbox、bgr to rgb、hwc to chw、normalize. 
    NOTE: the normalization coeffs need to be validated.
    */
    cudaStreamSynchronize(stream);
    STATS_START("detector inference:: image preprocess");
    preprocess(img, (float*)vBufferD[0], kInputH, kInputW, stream);
    cudaStreamSynchronize(stream);
    STATS_END("detector inference:: image preprocess");
    // 


    cudaStreamSynchronize(stream);
    STATS_START("detector inference:: model inference");
    
    // int64_t orig_target_sizes_h[2] = {srcWidth, srcHeight};
    // STATS_START("detector inference:: copy from host to device");
    int64_t orig_target_sizes_h[2] = {kInputW, kInputH};
    // CHECK(cudaMemcpyAsync(vBufferD[1], orig_target_sizes_h, 2 * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    // STATS_END("detector inference:: copy from host to device");

    bool status;
    for (int i = 0; i < numIO; i++)
    {
        status = context->setTensorAddress(IOnames[i].c_str(), vBufferD[i]);
    }
    // tensorrt inference
    // cudaStreamSynchronize(stream);
    // STATS_START("detector inference:: engine enqueue");
    context->enqueueV3(stream);
    // cudaStreamSynchronize(stream);
    // STATS_END("detector inference:: engine enqueue");

    
    // STATS_START("detector inference:: results copy");
    // The last three IOs are scores, boxes, and classes.
    CHECK(cudaMemcpyAsync(labels_h, vBufferD[numIO-3], OUTPUT_CANDIDATES * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(boxes_h, vBufferD[numIO-2], OUTPUT_CANDIDATES * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(scores_h, vBufferD[numIO-1], OUTPUT_CANDIDATES * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // cudaStreamSynchronize(stream);
    // STATS_END("detector inference:: results copy");


    // STATS_START("detector inference:: result postprocess");
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
    // STATS_END("detector inference:: result postprocess");


    cudaStreamSynchronize(stream);
    STATS_END("detector inference:: model inference");
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