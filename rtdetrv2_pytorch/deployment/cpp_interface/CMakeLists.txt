cmake_minimum_required(VERSION 3.10)
project(RTDETR VERSION 1.0)

# set cxx standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set debug mode
# set(CMAKE_BUILD_TYPE Release)
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# ====================== CUDA ======================
# enable CUDA as we have cu sources
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 89)
message(WARNING "set CMAKE_CUDA_ARCHITECTURES to sm89 to exclusively support 4090")
# we must set the above variable before enable CUDA
enable_language(CUDA)
set_ifndef(CUDA_INC_DIR /usr/local/cuda/include)
set_ifndef(CUDA_LIB_DIR /usr/local/cuda)
list(APPEND INCLUDE_DIR ${CUDA_INC_DIR})

# !!! PLEASE NOTE !!!
# @ink: The following command must precede the `add_subdirectory` command
# if we put it after the `add_subdirectory` command OR in the sub-CMakeLists,
# cmake could not find the cuda library correctly.
link_directories(/usr/local/cuda/lib64)

# ====================== OpenCV ======================
# this command defines OpenCV_LIBS and OpenCV_INCLUDE_DIRS
find_package(OpenCV)
list(APPEND INCLUDE_DIR ${OpenCV_INCLUDE_DIRS})
# list(APPEND INCLUDE_DIR "/usr/local/include/opencv4")
# list(APPEND INCLUDE_DIR ${OpenCV_CONTRIB_INCLUDE_DIRS})



# ====================== TensorRT ======================
# the following explicit include is not necessary, as the
# path is default path
# link_directories(/usr/lib/x86_64-linux-gpu) 

# basic infer lib
# find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
# set_ifndef(NVINFER_LIB ${_NVINFER_LIB})
# We need onnxparser
# find_library(_NVONNX_LIB nvonnxparser HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
# set_ifndef(NVONNX_LIB ${_NVONNX_LIB})

set(TENSORRT_LIBS
    nvinfer
    nvonnxparser
    # nvinfer_builder_resource
    # nvinfer_plugin
    # nvparsers
    # Add more as needed
)

set(INK_CUSTOM_PLUGIN_LIB_DIRS
    shared_plugin_libs
)


# ====================== third-party source ======================
add_subdirectory(third_party)
add_definitions(-DDEFINE_TRT_ENTRYPOINTS)

# ====================== user sources ======================
list(APPEND INCLUDE_DIR "./include")
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_library(usercustom SHARED ${SRCS})

message(WARNING "TRT_INCLUDE: ${TRT_INCLUDE} TRT_LIB: ${TRT_LIB}")
# message(WARNING "NVINFER_LIB: ${NVINFER_LIB}")
# message(WARNING "NVONNX_LIB: ${NVONNX_LIB}")
target_include_directories(usercustom PUBLIC ${INCLUDE_DIR})
target_include_directories(usercustom PUBLIC ${TRT_INCLUDE})
target_link_directories(usercustom PUBLIC ${TRT_LIB})
target_link_libraries(usercustom sample)
# target_link_libraries(usercustom ${NVINFER_LIB} ${NVONNX_LIB})
target_link_libraries(usercustom ${TENSORRT_LIBS})



add_executable(main main.cpp )
target_include_directories(main PUBLIC ${INCLUDE_DIR})
# include
target_include_directories(main PUBLIC ${TRT_INCLUDE})

# link
target_link_directories(main PUBLIC ${TRT_LIB} ${INK_CUSTOM_PLUGIN_LIB_DIRS})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ${TENSORRT_LIBS})
target_link_libraries(main usercustom)