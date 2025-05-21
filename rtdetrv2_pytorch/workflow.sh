# usage: workflow.sh [model]
set -e
set -x
# model must be specified
model=$1
# check if model is specified
if [ -z "$model" ]; then
    echo "Please specify output model name"
    exit 1
fi
# print model
echo "Model: $model"

# apply int8 quantization using `modelopt`library
python tools/quantization/export_onnx_mtq_fromPyTorch.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r benchmark_models/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --cali_set ~/workspace/coco_calib --output_file $model.onnx --simplify --check
# convert onnx to tensorrt engine using python api. `trtexec`is another way to do it.
python tools/export_trt.py --onnx $model.onnx --saveEngine $model.engine
# evaluation on COCO val2017
python tools/train.py -c  configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r benchmark_models/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth --test-only --trt-engine-path $model.engine --onnx-model-path $model.onnx