#!/usr/bin/env fish

# Check number of arguments
if count $argv | grep -q '^0$'
    echo "Usage: $0 <trt_version>"
    exit 1
end

set trt_version $argv[1]

# trt_version must be 10.7 or 10.9
if test "$trt_version" != "10.7" -a "$trt_version" != "10.9"
    echo "Invalid TensorRT version. Please use 10.7 or 10.9."
    exit 1
end

if test "$trt_version" = "10.7"
    echo "Using TensorRT version 10.7"
    set -x TRT_INCLUDE_PATH /root/workspace/TensorRT-v10.7.0.23-GA/include/
    set -x TRT_LIB_PATH /root/workspace/TensorRT-v10.7.0.23-GA/lib/
    set -x OPENCV_PATH /root/workspace/development/RTDETR_TensorRT_Deployment/opencv/build
    set -x LD_LIBRARY_PATH /root/workspace/TensorRT-v10.7.0.23-GA/lib/ $LD_LIBRARY_PATH
else
    echo "Using TensorRT version 10.9"
    set -x TRT_INCLUDE_PATH /root/workspace/TensorRT-v10.9.0.34-GA/include/
    set -x TRT_LIB_PATH /root/workspace/TensorRT-v10.9.0.34-GA/lib/
    set -x OPENCV_PATH /root/workspace/development/RTDETR_TensorRT_Deployment/opencv/build
    set -x LD_LIBRARY_PATH /root/workspace/TensorRT-v10.9.0.34-GA/lib/ $LD_LIBRARY_PATH
end
