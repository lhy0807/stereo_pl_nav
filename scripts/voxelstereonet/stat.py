from thop import profile, clever_format
from ptflops import get_model_complexity_info

from models.Voxel2D import Voxel2D
import torch
import torch.nn as nn

voxel_disp = []
for i in torch.arange(1,12):
    voxel_disp.append(torch.unsqueeze(i,0))

def input_constructor(input_shape):
    # For Flops-Counter method
    # Notice the input naming
    inputs = {'L': torch.randn(input_shape).cuda(), 'R': torch.randn(input_shape).cuda(), 'voxel_cost_vol':voxel_disp}
    return inputs

class WrappedModel(nn.Module):
	def __init__(self):
		super(WrappedModel, self).__init__()
		self.module = Voxel2D(192,"voxel") # that I actually define.
	def forward(self, *x):
		return self.module(*x)

model = WrappedModel()
model = model.cuda()
model.eval()
ckpt_path = "/home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs/lr_0.001_batch_size_16_cost_vol_type_voxel_optimizer_adam_Voxel2D_/best.ckpt"
print("Loading model {}".format(ckpt_path))
state_dict = torch.load(ckpt_path)
model.load_state_dict(state_dict['model'])

input_L = torch.randn(1, 3, 400, 880).cuda()
input_R = torch.randn(1, 3, 400, 880).cuda()

macs, params = profile(model, inputs=(input_L, input_R, voxel_disp))
macs, params = clever_format([macs, params], "%.3f")

print(f"MACs: {macs}, Params: {params}")

# macs, params = get_model_complexity_info(model, (1, 3, 400, 880), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True, input_constructor=input_constructor)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))