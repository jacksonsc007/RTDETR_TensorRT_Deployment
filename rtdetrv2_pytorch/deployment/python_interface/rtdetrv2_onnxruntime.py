"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torchvision.transforms as T

import numpy as np 
import onnxruntime as ort 
from PIL import Image, ImageDraw
import os
import time
from tqdm import tqdm


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for b in box:
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(lab[i].item()), fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    sess = ort.InferenceSession(args.onnx_file, providers=['CUDAExecutionProvider'])
    print(f"\033[96m[INFO] Run onnx model on {ort.get_device()}\033[0m")

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
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None]

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

        im_data = transforms(im_pil)[None]

        start = time.time()
        output = sess.run(
            # output_names=['labels', 'boxes', 'scores'],
            output_names=None,
            input_feed={'images': im_data.data.numpy(), "orig_target_sizes": orig_size.data.numpy()}
        )
        elapsed = time.time() - start

        labels, boxes, scores = output

        draw([im_pil], labels, boxes, scores, 0.4)
        # print("\033[96m[DEBUG] output:\033[0m")
        # import pprint
        # pprint.pprint(output)
        # print("\033[96m[DEBUG] max score:\033[0m")
        # pprint.pprint(output[-1].max())

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
    parser.add_argument('--onnx-file', type=str, )
    parser.add_argument('--im-file', type=str, )
    parser.add_argument('--img-dir', type=str, default='')
    # parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
