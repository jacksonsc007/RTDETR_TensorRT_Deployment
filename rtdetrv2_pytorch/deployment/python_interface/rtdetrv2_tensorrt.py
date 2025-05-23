"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import contextlib
import collections
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T 
import tensorrt as trt
import os

import sys
sys.path.append("src/solver")
from trt_engine import TRTInference
from tqdm import tqdm

import ctypes

# debug = True
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


def draw(images, labels, boxes, scores, prefix, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for l, b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(lab[l].item()), fill='blue', )
        output_dir = "benchmark_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"out_{prefix}_{i}.jpg")
        im.save(output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt-file', type=str, required=True)
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('--img-dir', type=str, default='')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('--plugin_c_lib', default='', type=str)
    parser.add_argument('--plugin_python', action='store_true', default=False)

    args = parser.parse_args()

    # load plugin
    if args.plugin_c_lib:
        """
        Explicitly register our custom plugins into custom namespace
        """
        # load plugin
        print("\n\033[91m[load c++ plugin]  \033[0m\n") 
        plugin_lib = ctypes.CDLL(args.plugin_c_lib)
        if not plugin_lib:
            raise RuntimeError("Could not load plugin library".format(args.plugin_c_lib))
        # Define the function prototype
        initFn = plugin_lib.initLibNvInferPlugins
        initFn.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        initFn.restype = ctypes.c_bool
        # Call the init function
        logger = trt.Logger(trt.Logger.VERBOSE)
        success = initFn(None, b"ink_plugins")
        if not success:
            raise RuntimeError("Failed to initialize ink_plugins")
    elif args.plugin_python:
        print("\n\033[92m[load python plugin]  \033[0m\n") 
        sys.path.append(".")
        import rtdetrv2_pytorch.ink_plugins.ink_plugins_IPluginV2
    else:
        print("\033[96m[INFO] No custom plugin libray is explicitly loaded\033[0m")

    m = TRTInference(args.trt_file, device=args.device)

    image_files = []
    if args.img_dir:
        for fname in os.listdir(args.img_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(args.img_dir, fname))
        image_files.sort()
    elif args.im_file:
        image_files = [args.im_file]
    else:
        raise ValueError("Either --im-file or --img-dir must be specified.")
    
    total_time_model = 0.0
    total_images = 0
    total_time_data_process = 0
    
    total_time_epoch = 0.0
    start_2 = time.time()
    for idx, img_path in enumerate(tqdm( image_files, desc="Processing Images")):
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        start_img_process= time.time()
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None]
        total_time_data_process += time.time() - start_img_process

        blob = {
            'images': im_data.to(args.device), 
            'orig_target_sizes': orig_size.to(args.device),
        }

        start = time.time()
        output = m(blob)
        elapsed = time.time() - start
        # print("\033[96m[DEBUG] output:\033[0m")
        # if idx % 100 == 0:
        #     import pprint
        #     # pprint.pprint(output)
        #     print("\033[96m[DEBUG] max score:\033[0m")
        #     pprint.pprint(output['scores'].max())
        #     draw([im_pil], output['labels'], output['boxes'], output['scores'], idx, 0.4) 
        # print(f"Processed {img_path} in {elapsed:.4f} seconds")
        total_time_model += elapsed
        total_images += 1
    total_time_epoch += time.time() - start_2

    result_dict = {
        'avg_time_model': total_time_model / total_images if total_images > 0 else 0,
        'avg_time_data_process': total_time_data_process / total_images if total_images > 0 else 0,
        'total_images': total_images,
        'fps (epoch)': total_images / total_time_epoch if total_images > 0 else 0,
        'fps (model)': total_images / total_time_model if total_images > 0 else 0,
    }
    if total_images > 0:
        def print_result(result_dict):
            for key, value in result_dict.items():
                print(f"{key}: {value:.4f}")
        print("\n\033[92m[INFO] Benchmark Result\033[0m")
        print("-" * 30)
        print_result(result_dict)