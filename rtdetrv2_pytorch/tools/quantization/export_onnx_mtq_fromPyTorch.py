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

from src.core import YAMLConfig

import modelopt.torch.quantization as mtq

from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image

debug = False

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
        

        
def track_layer_usage(model, dummy_input):
    used_layers = {}
    input_output_shape = {}
    
    def get_hook(name):
        def hook(module, input, output):
            used_layers[name] = True
            meta = {
                "input_shape": [_.shape for _ in input],
                "output_shape": [_.shape for _ in output],
                "module": module,
            }
            input_output_shape[name] = meta
            if isinstance(module, nn.Linear):
                print(name, meta)
            # if isinstance(module, nn.Conv2d):
            #     print(name, meta)
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

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().cuda()
    
    # NOTE: Find layers that are defined but does not join in forward pass
    data = torch.rand(1, 3, 640, 640).cuda()
    # data = ( data*255 - 128 ).round()
    size = torch.tensor([[640, 640]]).cuda()
    unused = track_layer_usage(model, (data, size))
    print("\033[96m[Warning]  The following layers are defined but not used in the forward pass:\033[0m")
    for layer, m in unused.items():
        print(f"Layer {layer}:{m} .")


    
    print(f"\033[96m[INFO] Preparing Calibration Dataset: \033[0m")
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
    _1 = model(data, size)
    def calibrate_loop(model):
        print("[INFO] Calibrate loop start...")
        for i, img in enumerate(dataloader):
            # print(f"\033[96m[INFO] Batch idx: {i} \033[0m")
            _ = model(img.cuda(), size)
        print("Done")
        
    # quant_cfg = mtq.FP8_DYNAMIC_CFG
    if args.q_type == 'int8':
        print("\033[92m[INFO] Apply int8 quantization \033[0m")
        # NOTE: default int8 quantization config
        quant_cfg = mtq.INT8_DEFAULT_CFG.copy()
                           
        """
        Do not apply quantization to unused layers.
        """
        # for layer, m in unused.items():
        #     quant_cfg['quant_cfg'][f'*{layer}*'] = {'enable': False}
            
                               
        
        # 1. Add specific quantization config for conv layers
        # NOTE: We do not delete '*weight_quantizer' in case of other types of weight_quantizer
        # quant_cfg['quant_cfg'].pop('*weight_quantizer')
        # quant_cfg['quant_cfg']['*conv*weight_quantizer'] = {
        #     'num_bits': 8,
        #     'axis': 0,
        #     'calibrator': 'max'
        # }
        # quant_cfg['quant_cfg']['*linear*weight_quantizer'] = {
        #     'num_bits': 8,
        #     'axis': 0,
        #     'calibrator': 'max'
        # }

        # quant_cfg['quant_cfg']['*conv*input_quantizer'] = {
        #     'num_bits': 8,
        #     'axis': None,
        #     'calibrator': 'max'
        # }

        # quant_cfg['quant_cfg']['*linear*input_quantizer'] = {
        #     'num_bits': 8,
        #     'axis': (1,),
        #     'calibrator': 'max'
        # }

        """
        specific config based on module type.
        """
        # quant_cfg['quant_cfg']['nn.Linear'] = {
        #     '*input_quantizer': {
        #         'num_bits': 8,
        #         'axis': (1,), # dealing with input size of (B, L, C)
        #         'calibrator': 'max',
        #         # 'calibrator': 'histogram', # only support per tensor quantization
        #         'fake_quant': True,
        #     }
        # }

        # quant_cfg['quant_cfg']['nn.Conv2d'] = {
        #     '*input_quantizer': {
        #         'num_bits': 8,
        #         'axis': None, # per tensor quantization
        #         'calibrator': 'histogram',
        #         'fake_quant': True,
        #     }
        # }

        """specific config for decoder.
        Note that the order of the following config matters. 
        """
        # 1.
        # quant_cfg['quant_cfg']['*decoder*input_quantizer'] = {
        #     'num_bits': 8,
        #     'axis': (1,),
        #     'calibrator': 'max'
        # }
        # quant_cfg['quant_cfg']['*decoder*input_proj*'] = {'enable': False}

        # 2.
        # quant_cfg['quant_cfg']['nn.Linear'] = {
        #     '*decoder*input_quantizer': {
        #         'num_bits': 8,
        #         'axis': (1,), # dealing with input size of (B, L, C)
        #         'calibrator': 'max',
        #         'fake_quant': True,
        #     }
        # }

        # quant_cfg['quant_cfg']['nn.Conv2d'] = {
        #     '*decoder*input_quantizer': {
        #         'num_bits': 8,
        #         'axis': None, # per tensor quantization
        #         'calibrator': 'histogram',
        #         'fake_quant': True,
        #     }
        # }
        
        
        # 3. decoder per layer quantization config
        # quant_cfg['quant_cfg']['nn.Linear'] = {
        #     '*decoder*layers.1.cross_attn.value_proj*input_quantizer': {
        #         'num_bits': 8,
        #         'axis': (1,), # dealing with input size of (B, L, C)
        #         'calibrator': 'max',
        #         'fake_quant': True,
        #     }
        # }
        # quant_cfg['quant_cfg']['nn.Linear'] = {
        #     '*decoder*head*input_quantizer': {
        #         'num_bits': 8,
        #         'axis': (1,), # dealing with input size of (B, L, C)
        #         'calibrator': 'max',
        #         'fake_quant': True,
        #     }
        # }
        
        
        # 4. disble particular units in decoder
        # quant_cfg['quant_cfg']['*attn*'] = {'enable': False}
        # quant_cfg['quant_cfg']['*cross_attn*value_proj*'] = {'enable': False}
        # quant_cfg['quant_cfg']['*cross_attn*output_proj*'] = {'enable': False}
        # quant_cfg['quant_cfg']['*cross_attn*attention_weights*'] = {'enable': False}
        # quant_cfg['quant_cfg']['*cross_attn*attention_weights*'] = {'enable': False}
        # quant_cfg['quant_cfg']['*cross_attn*sampling_offsets*'] = {'enable': False}
        
        
        # 5. what is fake quantization?
        # quant_cfg['quant_cfg']['*cross_attn*attention_weights*weight_quantizer'] = {
        #     'fake_quant': False,
        #     'num_bits': 8,
        #     'axis': 0
        # }
        # quant_cfg['quant_cfg']['*cross_attn*attention_weights*input_quantizer'] = {
        #     'fake_quant': False,
        #     'num_bits': 8,
        #     'axis': None
        # }

        
        # 6. modify the default quantization config
        # quant_cfg['quant_cfg']['*backbone*'] = {'enable': False}
        # quant_cfg['quant_cfg']['*encoder*'] = {'enable': False}
        # quant_cfg['quant_cfg']['*decoder*'] = {'enable': False}
        # quant_cfg['quant_cfg']['*model.decoder.input_proj*'] = {'enable': False}
        # quant_cfg["quant_cfg"]["*.bmm.output_quantizer"] = {
        #     "enable": False
        # }
        
        # 7. modify encoder quantization config
        
        # quant_cfg['quant_cfg']['nn.Linear'] = {
        #     '*encoder*input_quantizer': {
        #         'num_bits': 8,
        #         'axis': (1,), # dealing with input size of (B, L, C)
        #         'calibrator': 'max',
        #         'fake_quant': True,
        #     }
        # }

        # quant_cfg['quant_cfg']['nn.Conv2d'] = {
        #     '*input_quantizer': {
        #         'num_bits': 8,
        #         'axis': None, # per tensor quantization
        #         'calibrator': 'histogram',
        #         'fake_quant': True,
        #     }
        # }
        
    elif args.q_type == 'fp8':
        print("\033[92m[INFO] Apply fp8 quantization \033[0m")
        # quant_cfg = mtq.FP8_DYNAMIC_CFG.copy()
        # quant_cfg = mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG.copy()
        quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
        quant_cfg['quant_cfg']['nn.Conv2d'] = {
            '*backbone*': {
                'enable': False,
            }
        }
    elif args.q_type == 'int4':
        print("\033[92m[INFO] Apply int4 quantization \033[0m")
        # quant_cfg = mtq.FP8_DYNAMIC_CFG.copy()
        quant_cfg = mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG.copy()
    
    # quant_cfg["quant_cfg"]["*input_quantizer"]["num_bits"] = 32
    print("\033[96m[INFO] Quantizing model ... \033[0m")
    print(f"\033[96m[INFO] Quantization configuration: \033[0m")
    import pprint
    pprint.pprint(quant_cfg)
    """NOTE: 
    mtq.quantizea provokes error for `histogram calibrator`, so we do the calibration manually,
    following https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/63#:~:text=here%20is%20an%20example%20code
    """
    model = mtq.quantize(model, quant_cfg, calibrate_loop)
    # from modelopt.torch.quantization.model_calib import enable_stats_collection, finish_stats_collection
    # from modelopt.torch.quantization.model_quant import apply_mode
    # model = apply_mode(model, mode=[("quantize", quant_cfg)])
    # enable_stats_collection(model)
    # calibrate_loop(model)
    # finish_stats_collection(model, method="percentile")
    
    # NOTE: IMPORTANT! Manually move the model to cuda after calibration, or causing error
    model.cuda()   
    mtq.print_quant_summary(model)
    model.eval()

    # a pesue forward pass to test correctness
    _2 = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    print("\033[96m[INFO] Exporting to onnx model \033[0m")
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
    parser.add_argument('--q_type',  type=str, default='int8')

    args = parser.parse_args()

    main(args)
