from thop import profile, clever_format
from ptflops import get_model_complexity_info

from models.MSNet3D import MSNet3D
import torch
import torch.nn as nn

def input_constructor(input_shape):
    # For Flops-Counter method
    # Notice the input naming
    inputs = {'L': torch.ones(input_shape), 'R': torch.ones(input_shape)}
    return inputs

model = MSNet3D(192)

input_L = torch.randn(1, 3, 400, 880)
input_R = torch.randn(1, 3, 400, 880)

macs, params = profile(model, inputs=(input_L, input_R))
macs, params = clever_format([macs, params], "%.3f")

print(f"MACs: {macs}, Params: {params}")

macs, params = get_model_complexity_info(model, (1, 3, 400, 880), as_strings=True,
                                           print_per_layer_stat=True, verbose=True, input_constructor=input_constructor)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))