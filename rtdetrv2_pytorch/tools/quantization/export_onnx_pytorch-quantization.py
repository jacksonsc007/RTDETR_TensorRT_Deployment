"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os 
import sys
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import torch
import torch.nn as nn
import torchvision.transforms as T
import tqdm
import copy

from torchvision.io import read_image
from PIL import Image
from src.core import YAMLConfig
from torch.utils.data import Dataset

import pytorch_quantization
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import tensor_quant

debug = False
# debug = True
if debug:
    # improve torch tensor printing
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

    # debug with debugpy
    import debugpy
    # Listen on a specific port (choose any available port, e.g., 61074)
    debugpy.listen(("0.0.0.0", 61074))
    print("Waiting for debugger to attach...")
    # Optional: Wait for the debugger to attach before continuing execution
    debugpy.wait_for_client()

class CalibrationDateset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(self.img_dir) if (os.path.isfile(
            os.path.join(self.img_dir, f)) and (f.endswith('jpg')))
        ]
    def __len__(self, ):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        # print("[INFO] Loading image %d : %s" % (idx, img_path))
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        img = self.transform(img)
        return img

def collect_stats(model, data_loader, size):
    """Feed data to the network and collect statistic"""
    # Enable calibrators
    print("[INFO] collecting stats ... ")
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # for i, (image) in tqdm(enumerate(data_loader), total=num_batches):
    for i, (image) in enumerate(data_loader):
        _ = model(image.cuda(), size.cuda())

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
            
def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                print(f"[INFO] loading amax for {name}  ", end=" ")
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    # NOTE: we set strict to True, because some conv layers are not fed input in fpn_blocks. Dirty Code
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(**kwargs)
#             print(F"{name:40}: {module}")
    model.cuda()
        

def track_layer_usage(model, dummy_input):
    used_layers = {}
    
    def get_hook(name):
        def hook(module, input, output):
            used_layers[name] = True
        return hook

    # Register hooks for all modules
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.linear.Linear):
            hooks.append(module.register_forward_hook(get_hook(name)))
    
    # Use dummy input to trigger forward pass
    model(*dummy_input)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Find layers not used
    unused_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.linear.Linear):
            if name not in used_layers:
                unused_layers[name] = module
    
    return unused_layers

def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(torch.nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
        
    ori_model = Model().eval()

    # NOTE: Find layers that are defined but does not join in forward pass
    data = torch.rand(1, 3, 640, 640)
    # data = ( data*255 - 128 ).round()
    size = torch.tensor([[640, 640]])
    unused = track_layer_usage(ori_model, (data, size))
    for layer, m in unused.items():
        print(f"\033[96m[W]  Layer {layer}:{m} is defined but not used in the forward pass.\033[0m")
    


    # ==============================================================================================
    # =================================== Configure Quantization ===================================
    # ==============================================================================================
    # model = Model().cuda()
    # quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_desc_input_conv = QuantDescriptor(
        calib_method='histogram',
        num_bits=8,
        fake_quant=True,
        axis=None, # per-tensor quantization
    )
    # Calibrator histogram collection only supports per tensor scaling
    quant_desc_weight_conv = QuantDescriptor(
        calib_method='max',
        num_bits=8,
        fake_quant=True,
        axis=0, # per-channel (per-conv-kernel) quantization
    )

    quant_desc_weight_linear = QuantDescriptor(
        calib_method='max',
        num_bits=8,
        fake_quant=True,
        axis=0, # per-row quantization for linear layer weight (of shape [out_c, in_c])
    )
    quant_desc_input_linear = QuantDescriptor(
        calib_method='max',
        num_bits=8,
        fake_quant=True,
        axis=(1,), # for input of shape (B, L, in_c), we quantize per-row
    )

    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input_conv)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight_conv)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input_linear)
    quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight_linear)
    # quant_modules.initialize()

    

    
    # NOTE: we need to explicit convert linear and conv to quantized version,
    # since the stupid module are not able to identify them and convert them
    q_model = copy.deepcopy(ori_model).eval()


    # quant_desc_input_conv = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    # quant_desc_input_conv.calib_method = "histogram"
    # quant_desc_input_conv_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    # quant_desc_input_conv_weight.calib_method = "histogram"

    with torch.no_grad():
        for name, module in ori_model.named_modules():
            # from less to more
            # if "backbone" in name or "encoder" in name:
            if "quantize all conv":
                # TODO: use two equivalent conv2d because of some potential bugs
                # Durning experiments, the following two conv type showed up.
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.modules.conv.Conv2d):
                    print(f"[INFO] explicitly convert conv2d [ {name} ] to quantized conv2d ... ", end=" ")
                    q_module = quant_nn.QuantConv2d(
                        module.in_channels, module.out_channels, module.kernel_size,
                        module.stride, module.padding, module.dilation, module.groups,
                        module.bias is not None, module.padding_mode,
                        quant_desc_input=quant_desc_input_conv,
                        quant_desc_weight=quant_desc_weight_conv,
                        # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                        # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL,
                        
                    )
                    # q_module.set_default_quant_desc_input(quant_desc_input)
                    q_module.weight.data.copy_(module.weight.data.detach())
                    q_module.weight.requires_grad = False
                    if module.bias is not None:
                        q_module.bias.data.copy_(module.bias.data.detach())
                        q_module.bias.requires_grad = False
                    # setattr(q_model, name, q_module)
                    _ = q_model.get_submodule(name) # make sure we get name right
                    q_model.set_submodule(name, q_module)
                    print(" Done")

                # TODO: use two equivalent conv2d because of some potential bugs
                # Durning experiments, the following two conv type showed up.
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.modules.linear.Linear):
                print(f"[INFO] explicitly convert linear [ {name} ] to quantized linear ... ", end=" ")
                q_module = quant_nn.QuantLinear(
                    module.in_features, module.out_features,
                    quant_desc_input=quant_desc_input_linear,
                    quant_desc_weight=quant_desc_weight_linear,
                )
                # register hoook to check its input, in order to verify the quantization axes for input
                # （bs, L, in_c)
                # q_module.register_forward_pre_hook(
                #     lambda module, input: print(f"input: {input[0].shape} {input[0].min()} {input[0].max()}")
                # )
                # q_module.set_default_quant_desc_input(quant_desc_input)
                q_module.weight.data.copy_(module.weight.data.detach())
                q_module.weight.requires_grad = False
                if module.bias is not None:
                    q_module.bias.data.copy_(module.bias.data.detach())
                    q_module.bias.requires_grad = False
                # setattr(q_model, name, q_module)
                q_model.set_submodule(name, q_module)
                print(" Done")


    pre_transforms = T.Compose([
        T.Resize(( 640,640 )),
        T.ToTensor()
    ])
    cali_dataset = CalibrationDateset(args.cali_set, transform=pre_transforms)
    dataloader = torch.utils.data.DataLoader(
        cali_dataset, batch_size=8, shuffle=False
    )
    
    # For quantized INT8 model, the input value range is quantized to [-128, 127]
    data = torch.rand(1, 3, 640, 640).cuda()
    # data = ( data*255 - 128 ).round()
    size = torch.tensor([[640, 640]]).cuda()
    model = q_model.cuda()
    _1 = model(data, size)

    model.eval()
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, dataloader, size)
        compute_amax(model, method="percentile", percentile=99.99, strict=False)

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print("="*100)
            print(name, module)
    _2 = model(data, size)



    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }
    
    # manually diable all grad
    for p in model.parameters():
        p.requires_grad = False


    with pytorch_quantization.enable_onnx_export():
        torch.onnx.export(
            model, 
            (data, size), 
            args.output_file,
            input_names=['images', 'orig_target_sizes'],
            output_names=['labels', 'boxes', 'scores'],
            dynamic_axes=dynamic_axes,
            opset_version=17, 
            verbose=False,
            do_constant_folding=True,
            onnx_checker=False
        )

    if args.check:
        import onnx
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx 
        import onnxsim
        dynamic = True 
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(onnx_model, input_shapes=input_shapes, dynamic_input_shape=dynamic)
        onnx.save(onnx_model_simplify, args.output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)
    parser.add_argument('--cali_set',  type=str, default=None)

    args = parser.parse_args()

    main(args)
