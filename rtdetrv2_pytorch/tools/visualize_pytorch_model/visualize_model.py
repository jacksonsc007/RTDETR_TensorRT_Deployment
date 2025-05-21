"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 

from src.core import YAMLConfig
import wandb
import torchexplorer

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

    BASE_MODEL = cfg.model.deploy()
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.backbone = BASE_MODEL.backbone
            self.encoder = BASE_MODEL.encoder
            
        def forward(self, images):
            x = self.backbone(images)
            outputs = self.encoder(x[0], x[1], x[2])
            return outputs

    model = Model() 
    model = model.train()
    # visualize model
    # torchexplorer.setup()
    # wandb.init()
    torchexplorer.watch(
        model, 
        backend="standalone",
        log = ['io', 'params'],
)

    data = torch.rand(1, 3, 640, 640)
    # size = torch.tensor([[640, 640]])
    _ = model(data)
    _.sum().backward()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)

    args = parser.parse_args()

    main(args)
