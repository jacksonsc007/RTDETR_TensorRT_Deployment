#!/usr/local/bin/bash
set -e
set -u
# check the number of arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <trt_version>"
    exit 1
fi
trt_version=$1
if [ "$trt_version" == "10.7" ]; then
    export TRT_INCLUDE_PATH=/root/workspace/TensorRT-v10.7.0.23-GA/include/
    export TRT_LIB_PATH=/root/workspace/TensorRT-v10.7.0.23-GA/lib/
    export OPENCV_PATH=/root/workspace/RTDETR_TensorRT_Deployment/opencv/build
    export LD_LIBRARY_PATH=/root/workspace/TensorRT-v10.7.0.23-GA/lib:$LD_LIBRARY_PATH
else
    export TRT_INCLUDE_PATH=/root/workspace/TensorRT-v10.9.0.34-GA/include/
    export TRT_LIB_PATH=/root/workspace/TensorRT-v10.9.0.34-GA/lib/
    export OPENCV_PATH=/root/workspace/RTDETR_TensorRT_Deployment/opencv/build
    export LD_LIBRARY_PATH=/root/workspace/TensorRT-v10.9.0.34-GA/lib:$LD_LIBRARY_PATH
fi