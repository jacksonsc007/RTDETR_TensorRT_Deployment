set -e

# check if those env variables are set
if [ -z "$TRT_INCLUDE_PATH" ] || [ -z "$TRT_LIB_PATH" ] || [ -z "$OPENCV_PATH" ]; then
    echo "Please set the following environment variables: TRT_INCLUDE_PATH, TRT_LIB_PATH, OPENCV_PATH"
    echo "TRT_INCLUDE_PATH: $TRT_INCLUDE_PATH"
    echo "TRT_LIB_PATH: $TRT_LIB_PATH"
    echo "OPENCV_PATH: $OPENCV_PATH"
    echo "run set_env.sh to set them"
    exit 1
fi
echo "TRT_INCLUDE_PATH: $TRT_INCLUDE_PATH"
echo "TRT_LIB_PATH: $TRT_LIB_PATH"
echo "OPENCV_PATH: $OPENCV_PATH"

# check if the directories exist
if [ ! -d "$TRT_INCLUDE_PATH" ]; then
    echo "TRT_INCLUDE_PATH does not exist: $TRT_INCLUDE_PATH"
    exit 1
fi
if [ ! -d "$TRT_LIB_PATH" ]; then
    echo "TRT_LIB_PATH does not exist: $TRT_LIB_PATH"
    exit 1
fi
if [ ! -d "$OPENCV_PATH" ]; then
    echo "OPENCV_PATH does not exist: $OPENCV_PATH"
    exit 1
fi

# NOTE: do not leave trailing space after backslash
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DTRT_INCLUDE="$TRT_INCLUDE_PATH" -DTRT_LIB="$TRT_LIB_PATH" \
    -DOpenCV_DIR="$OPENCV_PATH" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 

cmake --build build -j