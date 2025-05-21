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
import modelopt.onnx.quantization as moq

from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image

debug = True

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
        width, height = img.width, img.height
        # img = read_image(img_path)
        img = self.transform(img)
        # get width and height of the image
        return img, (width, height)
        

        

def main(args, ):
    """main
    """
    pre_transforms = T.Compose([
        T.Resize(( 640,640 )),
        T.ToTensor()
    ])
    cali_dataset = CalibrationDateset(args.cali_set, transform=pre_transforms)
    cali_imgs = []
    cali_size = []
    for idx, (img, img_size) in enumerate(cali_dataset):
        cali_imgs.append(img)
        cali_size.append(img_size)
    cali_imgs = torch.stack(cali_imgs, dim=0).numpy()
    size = torch.tensor(cali_size).numpy()
    print(f"\033[91m{cali_imgs.shape = }\033[0m")
    print(f"\033[91m{size.shape = }\033[0m")
    calib_data = {
        'images': cali_imgs,
        'orig_target_sizes': size
    }
    import numpy as np
    # np.savez("calib_data.npz", calib_data)
    moq.quantize(
        onnx_path=args.onnx_path,
        calibration_data=calib_data,
        output_path=args.output_file,
        quantize_mode="int8",
        calibration_method="max",
        nodes_to_quantize=[".*decoder.*"]
    )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', '-o', type=str, default=None)
    parser.add_argument('--cali_set',  type=str, default=None)
    parser.add_argument('--onnx_path',  type=str, default=None)

    args = parser.parse_args()

    main(args)
