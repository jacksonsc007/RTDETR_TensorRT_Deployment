#!/usr/local/bin/bash
set -e
set -u
# check the number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_name> <trt_version> <trtexec args>"
    exit 1
fi
onnx_model=$1
trt_version=$2
trtexec_args=${@:3}

# raise error if either model or trt_version is not provided
if [ -z "$onnx_model" ]; then
    echo "Usage: $0 <model_name> <trt_version> <trtexec args>"
    exit 1
fi

if [ -z "$trt_version" ]; then
    echo "Usage: $0 <model_name> <trt_version> <trtexec args>"
    exit 1
fi
if [ "$trt_version" != "10.7" ] && [ "$trt_version" != "10.9" ]; then
    echo "Invalid TensorRT version. Please use 10.7 or 10.9."
    exit 1
fi
# when version is 10.7
if [ "$trt_version" == "10.7" ]; then
    trtexec_bin=/root/workspace/RTDETR_TensorRT_Deployment/TensorRT-v10.7/out/trtexec
    output_dir=tensorrt_10_7/${onnx_model}-${trtexec_args}
    export LD_LIBRARY_PATH=/root/workspace/TensorRT-v10.7.0.23-GA/lib:$LD_LIBRARY_PATH
    export PATH=/root/workspace/RT-DETR/TensorRT-v10.7/out:$PATH # specify the path to trtexec
else
    trtexec_bin=/root/workspace/RTDETR_TensorRT_Deployment/TensorRT-v10.9/out/trtexec
    output_dir=tensorrt_10_9/${onnx_model}-${trtexec_args}
    export LD_LIBRARY_PATH=/root/workspace/TensorRT-v10.9.0.34-GA/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
    export PATH=/root/workspace/RT-DETR/TensorRT-v10.9/out:$PATH # specify the path to trtexec
fi

if [ ! -d "$output_dir" ]; then
    mkdir -p $output_dir
fi

python ../../TensorRT-v10.7/tools/experimental/trt-engine-explorer/utils/process_engine.py  ${onnx_model} ${output_dir} ${trtexec_args}