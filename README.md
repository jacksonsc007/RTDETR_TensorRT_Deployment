## Introduction
This repository intends to provide a comprehensive guide to deploy RT-DETR with TensorRT, based on both Python and C++.

## Development Environment
### C++ Environment
- OpenCV branch: `4.x`
- TensorRT: 10.9.0.34 or 10.7.0.23

#### Build OpenCV
Follow the instructions in [OpenCV](https://docs.opencv.org/4.11.0/d7/d9f/tutorial_linux_install.html) for details to build OpenCV.
Here is a simple but workable guide:
```fish
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
```

#### Build TensorRT

TensorRT is partial open-sourced. To build it, we need to download both open-sourced part (Officially named  **TensorRT-OSS components**) and closed-sourced part (**TensorRT GA** ). To clarify:
- **TensorRT-OSS components**: https://github.com/NVIDIA/TensorRT
```fish
git clone -b v10.7.0 main https://github.com/nvidia/TensorRT TensorRT-v10.7
cd TensorRT-v10.7
git submodule update --init --recursive
```

If TensorRT-v10.7 is intended to be used, feel free to use the existing `TensorRT-v10.7` directory on this repositoty, where a new sample is added to test cumstom plugins.

- **TensorRT GA**: [download and extract the TensorRT GA build ](https://github.com/NVIDIA/TensorRT#:~:text=download%20and%20extract%20the%20tensorrt%20ga%20build%20)
![[Pasted image 20250512103628.png]]

> TensorRT GA consists of all necessary components to build our project with TensorRT. The primary objective of building with TensorRT-OSS is to generate `trtexec`, which offers command-line utilities.

- After finishing downloading, enter the directory of TensorRT-OSS and run the following commands:
```shell
# fish shell
[2025-05-12-10:41]    set TRT_GA_LIBPATH ~/workspace/TensorRT-v10.7.0.23-GA/lib
[2025-05-12-10:41]    set TRT_OSS_PATH ~/workspace/RTDETR_TensorRT_Deployment/TensorRT-v10.7/
[2025-05-12-10:41]    cd $TRT_OSS_PATH 
[2025-05-12-10:41]    cmake -B build -DTRT_LIB_DIR=$TRT_GA_LIBPATH -DTRT_OUT_DIR=$(pwd)/out -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build -j

```

### Python Environment
```fish
uv sync
```

## Usages
### Quick workflow
Apply the workflow of quantization, conversion and evaluation.
```fish
bash workflow.sh <output_model_name>
```

### Manually Fuse Nodes in the ONNX Model
For RT-DETR, TensorRT adopts an inappropriate fusion tactic which leads to severe precision degradation during converting quantized ONNX model to TensorRT engine.

To prevent TensorRT from performing this, we manually fuse ONNX nodes that are erroneously fused, before converting into TensorRT engine.
```fish
source .venv/bin/activate.fish
cd rtdetrv2_pytorch
python tools/modify_onnx_model/graph_surgeon_rtdetr.py --onnx_file benchmark_models/default_mtq_int8_q_qint8.onnx --output_file fused_attn_sp-default_mtq_int8_q_qint8.onnx
```

### Custom Plugin Implementation
TensorRT is incapable of finding an appropriate tactic for our fused nodes, so we need to implement a custom plugin to handle the fused nodes.
#### C++ Plugin Implementation
C++ Plugin Implementation can be used in both Python and C++, as the plugins are built into a shared library.

Two implementations are developed based on C++ API:
- rtdetrv2_pytorch/ink_plugins/ink_plugins_v2.cu
- rtdetrv2_pytorch/ink_plugins/ink_plugins_v3.cu

To build into a shared library, run the following command:
```fish
cd rtdetrv2_pytorch/ink_plugins/
. set_env.fish 10.9
bash build_ink_plugin.sh 10.9
```

To use the built shared library in Python:
```python
# rtdetrv2_pytorch/tools/export_trt.py 

for lib in plugin_libs:
  builder.get_plugin_registry().load_library(lib)
```

To use the built shared library in C++:
```C++
// rtdetrv2_pytorch/deployment/cpp_interface/src/rtdetr.cpp
  if (runtime->getPluginRegistry().loadLibrary(pluginLibs.c_str()) == nullptr)
  {
      printf("\e[31m[ERROR]\e[m Failed loading plugin library!\n");
      return -1;
  };
```

#### Python Plugin Implementation
Python Plugin Implementation can only be used in Python.
Three implementations were developed based on Python API:
- rtdetrv2_pytorch/ink_plugins/ink_plugins_decorator.py
- rtdetrv2_pytorch/ink_plugins/ink_plugins_IPluginV2.py
- rtdetrv2_pytorch/ink_plugins/ink_plugins_IPluginV3.py

To use it, we should explicitly import them in our python scripts:
```python
# import custom plugins
import sys
sys.path.append(".")
# Load if we intend to use python plugins
# import ink_plugins.ink_plugins_decorator 
# import ink_plugins.ink_pluginsIPluginV2 
import ink_plugins.ink_plugins_IPluginV3 
```
and then we can use them in our python scripts:
```fish
source .venv/bin/activate.fish

# create engine
python tools/export_trt.py --onnx tools/modify_onnx_model/fused_attn_sp-default_mtq_int8_q_qint8.onnx --saveEngine=fused_attn_sp-default_mtq_int8_q_qint8.engine

# evaluation
python tools/train.py -c  configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r benchmark_models/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --test-only --trt-engine-path fused_attn_sp-default_mtq_int8_q_qint8.engine
```
> Plugins based on decorator is buggy and did not work well at the time of writting. The others are recommended.


### TensorRT Deployment Based on C++
```fish
[2025-05-20-14:24]    cd deployment/cpp_interface/
[2025-05-20-14:26]    . set_env.fish 10.9
[2025-05-20-14:27]    bash build.sh
[2025-05-20-14:28]    build/main --image_dir ../../dataset/benchmark_fps_dataset/ --onnx_file ../../benchmark_models/fused_attn_sp-default_mtq_int8_q_qint8.onnx
[2025-05-20-14:29]    build/main --image_dir ../../dataset/benchmark_fps_dataset/ --trt_file built_engine/default_built.engine
```
- Make sure the TensorRT library path is set correctly in the set_env.fish script.
- To switch TensorRT version, please modify the arguments of set_env.fish. And modify the pluginlib path in `RTdetr` class in `rtdetrv2_pytorch/deployment/cpp_interface/include/rtdetr.h`.

### TensorRT Deployment Based on Python
Python deployment is straightforward:
```fish
source .venv/bin/activate.fish
python deployment/python_interface/rtdetrv2_tensorrt.py  --img-dir dataset/benchmark_fps_dataset/ --trt benchmark_models/tensorrt_10_7/default_mtq_int8_q_qint8.onnx-best/default_mtq_int8_q_qint8.onnx.engin
```

here is a test

### Minimal C++ Plugin Implementation And Test Scripts
A minimal cpp plugin implementation and test scripts are provided in `TensorRT-v10.7/samples/ink_plugins` folder.

To use it, build it along with TensorRT:
```fish
cd TensorRT-v10.7
cmake -B build -DTRT_LIB_DIR=$TRT_GA_LIBPATH -DTRT_OUT_DIR=$(pwd)/out -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build -j
```

### Profile and Visualize Engine
```fish
source .venv/bin/activate.fish
cd rtdetrv2_pytorch/benchmark_models/
bash build_profile_visualize_engine.sh default_mtq_int8_q_qint8.onnx 10.7 best
```

## Acknowledgements
- RT-DETR: https://github.com/lyuwenyu/RT-DETR
- TensorRT-YOLO11: https://github.com/emptysoal/TensorRT-YOLO11
- TensorRT: https://github.com/NVIDIA/TensorRT
- OpenCV: https://github.com/opencv/opencv