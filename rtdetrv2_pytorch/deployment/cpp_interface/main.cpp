#include "utils.h"
// #include "infer.h"
#include "rtdetr.h"
#include <filesystem> 
#include "profile.h"


int run(const char* imageDir, const char* trtFile, const char* onnxFile){
    // get image file names for inferencing
    std::vector<std::string> file_names;
    if (read_files_in_dir(imageDir, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    // create detector, and load engine plan
    // std::string trtFile = "./yolo11s.plan";
    // YoloDetector detector(trtFile);
    // std::string trtFile = "./rtdetrv2_r18vd_120e_coco.engine";

    RTDetr detector(trtFile, onnxFile);
    // inference
    auto start_epoch = std::chrono::system_clock::now();
    double model_inference_time = 0.0;
    for (long unsigned int i = 0; i < file_names.size(); i++){
        STATS_START("read image");
        std::string imagePath = std::string(imageDir) + "/" + file_names[i];
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (img.empty()) continue;
        STATS_END("read image");

        auto start = std::chrono::system_clock::now();
        
        STATS_START("detector inference");
        std::vector<Detection> res = detector.inference(img);
        STATS_END("detector inference");

        auto end = std::chrono::system_clock::now();
        int cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double inference_time = std::chrono::duration<double, std::milli>(end - start).count();
        model_inference_time += inference_time;

        // std::cout << "Image: " << file_names[i] << " cost: " << cost << " ms."  << std::endl;

        // draw result on image
        bool draw_and_save = false;
        if (draw_and_save)
        {
            RTDetr::draw_image(img, res);

            std::string output_dir = "./output";
            if (!std::filesystem::exists(output_dir)) {
                std::filesystem::create_directory(output_dir);
            }
            cv::imwrite(output_dir + "/" + file_names[i], img);

            // std::cout << "Image: " << file_names[i] << " done." << std::endl;
        }
    }
    auto end_epoch = std::chrono::system_clock::now();
    double epoch_time = std::chrono::duration<double, std::milli>(end_epoch - start_epoch).count();
    double fps_epoch = file_names.size() / (epoch_time / 1000.0);
    double fps_model_inference = file_names.size() / (model_inference_time / 1000.0);
    printf("\e[31m[INFO]\e[m RT-DETR FPS: %.2f %.2f\n", fps_epoch, fps_model_inference);
    printf("\e[31m[INFO]\e[m RT-DETR Deteciton Finished.\n");
    REPORT
    return 0;
}


int main(int argc, char *argv[])
{
    /*
    Usage:
    ./main --image_dir [image dir] --onnx_file [onnx file]
    ./main --image_dir [image dir] --trt_file [trt file] 
    Example:
    ./main --image_dir ./images --onnx_file ./rtdetrv2_r18vd_120e_coco.onnx
    ./main --image_dir ./images --trt_file ./rtdetrv2_r18vd_120e_coco.engine
    */
    // Default values
    std::string imageDir = "";
    std::string trtFile = "";
    std::string onnxFile = "";

    if (argc == 1) {
        std::cout << "Usage:" << std::endl;
        std::cout << "./main --image_dir [image dir] --onnx_file [onnx file]" << std::endl;
        std::cout << "./main --image_dir [image dir] --trt_file [trt file]" << std::endl;
        return -1;
    }
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--image_dir" && i + 1 < argc) {
            imageDir = argv[++i];
        } else if (arg == "--trt_file" && i + 1 < argc) {
            trtFile = argv[++i];
        } else if (arg == "--onnx_file" && i + 1 < argc) {
            onnxFile = argv[++i];
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
            std::cout << "Usage:" << std::endl;
            std::cout << "./main --image_dir [image dir] --onnx_file [onnx file]" << std::endl;
            std::cout << "./main --image_dir [image dir] --trt_file [trt file]" << std::endl;
            return -1;
        }
    }

    // Validate arguments
    if (imageDir.empty()) {
        std::cout << "Error: Image directory not specified" << std::endl;
        return -1;
    }

    if (trtFile.empty() && onnxFile.empty()) {
        std::cout << "Error: Either trt_file or onnx_file must be specified" << std::endl;
        return -1;
    }
    
    // Call run function with parsed arguments
    return run(imageDir.c_str(), trtFile.c_str(), onnxFile.c_str());
}
