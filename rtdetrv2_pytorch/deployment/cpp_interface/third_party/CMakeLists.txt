# utilize the sample lib in the official tensorrt repository
file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)
add_library(sample SHARED ${SRCS})

# cuda
# find_package(CUDA REQUIRED)
message(STATUS "cuda libraries: ${CUDA_LIBRARIES}")

list(APPEND INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common ${CMAKE_CURRENT_SOURCE_DIR}/utils)

# tensorrt
# find_package(TensorRT REQUIRED)
# include_directories(${TensorRT_INCLUDE_DIRS})

# message(STATUS "Thirdparty: TRT_INCLUDE: ${TRT_INCLUDE} TRT_LIB: ${TRT_LIB}")
target_include_directories(sample PUBLIC ${INCLUDE_DIR})
target_include_directories(sample PUBLIC ${TRT_INCLUDE})

target_link_directories(sample PUBLIC ${TRT_LIB})
target_link_libraries(sample cudart)
message(WARNING "TensorRT_LIBS: ${TENSORRT_LIBS}")
target_link_libraries(sample ${TENSORRT_LIBS})
# target_link_libraries(sample ${TensorRT_LIBRARIES})