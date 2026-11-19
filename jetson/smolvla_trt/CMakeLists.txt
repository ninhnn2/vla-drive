# vla_probe: run the fine-tuned SmolVLA (smolvla-v1) with TensorRT, and show each stage.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#
# On the Jetson (JetPack 6) TensorRT, CUDA and OpenCV are where CMake looks by default.
# Elsewhere point TENSORRT_ROOT at a TensorRT install: -DTENSORRT_ROOT=/opt/tensorrt
cmake_minimum_required(VERSION 3.18)
project(vla_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT install prefix (empty: system paths)")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
          HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)
find_library(NVINFER_LIBRARY nvinfer
             HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIBRARY)
  message(FATAL_ERROR "TensorRT not found; set -DTENSORRT_ROOT=<prefix>")
endif()
message(STATUS "TensorRT: ${TENSORRT_INCLUDE_DIR}, ${NVINFER_LIBRARY}")

add_executable(vla_probe
  src/main.cpp
  src/smolvla.cpp
  src/tokenizer.cpp
  src/preprocess.cpp
  src/trt_engine.cpp
  src/meta.cpp)
target_include_directories(vla_probe SYSTEM PRIVATE ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vla_probe PRIVATE ${NVINFER_LIBRARY} CUDA::cudart ${OpenCV_LIBS})
target_compile_options(vla_probe PRIVATE -Wall -Wextra -Wno-unused-parameter)
