from thop import profile, clever_format
from ptflops import get_model_complexity_info

from networks.stackhourglass import PSMNet
import torch
import torch.nn as nn

def input_constructor(input_shape):
    # For Flops-Counter method
    # Notice the input naming
    inputs = {'L': torch.ones(input_shape), 'R': torch.ones(input_shape)}
    return inputs

affinity_settings = {}
affinity_settings['win_w'] = 3
affinity_settings['win_h'] = 3
affinity_settings['dilation'] = [1, 2, 4, 8]

model = PSMNet(maxdisp=192, struct_fea_c=4, fuse_mode='separate',
               affinity_settings=affinity_settings, udc=True, refine='csr')
model.eval()

input_L = torch.randn(1, 3, 400, 880, device="cpu")
input_R = torch.randn(1, 3, 400, 880, device="cpu")
input_gt = torch.randn(1, 1, 400, 880, device="cpu")

macs, params = profile(model, inputs=(input_L, input_R, None))
macs, params = clever_format([macs, params], "%.3f")

print(f"MACs: {macs}, Params: {params}")