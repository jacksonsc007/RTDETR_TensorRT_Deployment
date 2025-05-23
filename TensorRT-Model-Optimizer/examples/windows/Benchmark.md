# TensorRT Model Optimizer - Windows: Benchmark Reference

This document provides a summary of the performance and accuracy measurements of [TensorRT Model Optimizer - Windows](./README.md) for several popular models. The benchmark results in the following tables serve as reference points and **should not be viewed as the maximum performance** achievable by Model Optimizer - Windows.

### 1 Performance And Accuracy Comparison: ONNX INT4 vs ONNX FP16 Models

#### 1.1 Performance Comparison

All performance metrics are tested using the [onnxruntime-genai perf benchmark](https://github.com/microsoft/onnxruntime-genai/tree/main/benchmark/python) with the DirectML backend.

- **Configuration**: Windows OS, GPU RTX 4090, NVIDIA Model Optimizer v0.19.0.
- **Batch Size**: 1

Memory savings and inference speedup are compared to the ONNX FP16 baseline.

| | | | | |
|:------------------------|:------------------------|:-------------------------|:----------------------|:-----------------------------------------|
| **Model** | **Input Prompt Length** | **Output tokens length** | **GPU Memory Saving** | **Generation Phase Inference Speedup** |
|[Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 128 | 256 | 2.44x | 2.68x |
|[Phi3.5-mini-Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 128 | 256 | 2.53x | 2.51x |
|[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 128 | 256 | 2.88x | 3.41x |
|[Llama3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 128 | 256 | 1.96x | 2.19x |
|[Gemma-2b-it](https://huggingface.co/google/gemma-2b-it) | 128 | 256 | 1.64x | 1.94x |

#### 1.2 Accuracy Comparison

For accuracy evaluation, the [Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) benchmark has been utilized. Please refer to the [detailed instructions](./accuracy_benchmark/README.md) for running the MMLU accuracy benchmark.

The table below shows the MMLU 5-shot score for some models.

- **FP16 ONNX model**: Generated using [GenAI Model Builder](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/README.md) with DML EP
- **INT4 AWQ model**: Generated by quantizing FP16 ONNX model using ModelOpt-Windows
- **Configuration**: Windows OS, GPU RTX4090, nvidia-modelopt v0.19.0, onnxruntime-genai-directml 0.4, transformers 4.44

| **Model** | **ONNX FP16** | **ONNX INT4** |
|:------------------------------|:---------------|:--------------|
| [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 68.45 | 66.1 |
| [Phi3.5-mini-Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 68.9 | 65.7 |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 61.76 | 60.73 |
| [Llama3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 60.8 | 57.71 |
| [Gemma-2b-it](https://huggingface.co/google/gemma-2b-it) | 37.01 | 37.2 |
