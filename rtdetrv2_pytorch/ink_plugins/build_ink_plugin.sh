set -e

# check if those env variables are set
if [ -z "$TRT_INCLUDE_PATH" ] || [ -z "$TRT_LIB_PATH" ]; then
    echo "Please set the following environment variables: TRT_INCLUDE_PATH, TRT_LIB_PATH, OPENCV_PATH"
    echo "TRT_INCLUDE_PATH: $TRT_INCLUDE_PATH"
    echo "TRT_LIB_PATH: $TRT_LIB_PATH"
    echo "run set_env.sh to set them"
    exit 1
fi

trt_version=$1
if [ -z "$trt_version" ]; then
    echo "Usage: $0 <trt_version>"
    exit 1
fi


# There are only two versions available: 10.7 and 10.9
if [ "$trt_version" != "10.7" ] && [ "$trt_version" != "10.9" ]; then
    echo "Invalid TensorRT version. Please use 10.7 or 10.9."
    exit 1
fi

# when version is 10.7
if [ "$trt_version" == "10.7" ]; then
    echo "Building ink plugin for TensorRT 10.7"
    # Set the path to the TensorRT 10.7 installation
    OUTPUT_DIR=tensorrt_10_7
else
    echo "Building ink plugin for TensorRT 10.9"
    # Set the path to the TensorRT 10.9 installation
    OUTPUT_DIR=tensorrt_10_9
fi

if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

echo "Building ink plugin for TensorRT ${trt_version}"
rm -rf $OUTPUT_DIR
cmake -B $OUTPUT_DIR -DTRT_INCLUDE=${TRT_INCLUDE_PATH} -DTRT_LIB=${TRT_LIB_PATH}  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build $OUTPUT_DIR -j
