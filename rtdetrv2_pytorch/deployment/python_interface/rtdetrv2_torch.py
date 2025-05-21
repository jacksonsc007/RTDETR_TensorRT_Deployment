"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw
import os
import time

import sys
sys.path.append(".")
from src.core import YAMLConfig
from tqdm import tqdm


def draw(images, labels, boxes, scores, thrh = 0.6, out_prefix='results'):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'{out_prefix}_{i}.jpg')


def main(args):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

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

    total_time = 0.0
    total_images = 0

    # time including data preprocess
    total_time_2 = 0.0
    start_2 = time.time()
    # for idx, img_path in enumerate(image_files):
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing Images")):
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        im_data = transforms(im_pil)[None].to(args.device)

        torch.cuda.synchronize() if args.device.startswith('cuda') else None
        start = time.time()
        with torch.no_grad():
            output = model(im_data, orig_size)
        torch.cuda.synchronize() if args.device.startswith('cuda') else None
        elapsed = time.time() - start

        labels, boxes, scores = output
        draw([im_pil], labels, boxes, scores, out_prefix=f'results_{idx}')
        # print(f"Processed {img_path} in {elapsed:.4f} seconds")
        total_time += elapsed
        total_images += 1

    total_time_2 += time.time() - start_2
    if total_images > 0:
        fps = total_images / total_time
        fps_2 = total_images / total_time_2
        print(f"\nProcessed {total_images} images in {total_time:.2f} seconds. FPS: {fps:.2f} FPS_2: {fps_2:.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('--img-dir', type=str, default='')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
