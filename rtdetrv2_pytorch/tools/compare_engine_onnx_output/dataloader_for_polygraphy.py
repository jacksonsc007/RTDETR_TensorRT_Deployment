"""
This script generates data for polygraphy command line usage.
"""
import os 
import sys

import torch.utils
import torch.utils.data
import torch.utils.data.dataloader 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 

from src.core import YAMLConfig


from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image
import numpy as np


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
        return img, (width, height)
        

        

pre_transforms = T.Compose([
    T.Resize(( 640,640 )),
    T.ToTensor()
])
cali_set = "/workspace/trt_projects/TensorRT-RTDETR/C++/detect/images"
cali_dataset = CalibrationDateset(cali_set, transform=pre_transforms)
dataloader = torch.utils.data.DataLoader(
    cali_dataset, batch_size=1, shuffle=False
)

data_iterator = iter(dataloader)

size = torch.tensor([[640, 640]]).numpy()
    
    
def load_data():
    for _ in range(5):
        image, size = next(data_iterator)
        yield {
            "images": image.numpy(),
            'orig_target_sizes': np.array(size)
        }  # Still totally real data
        
input_data = list(load_data())
print(input_data)