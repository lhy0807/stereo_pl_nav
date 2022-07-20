from thop import profile, clever_format
from ptflops import get_model_complexity_info
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from models.Voxel2D import Voxel2D
from models.Voxel2D_sparse import Voxel2D
# from models.Voxel2D_hie import Voxel2D
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from datasets import VoxelDSDataset
from tqdm import tqdm
from spconv.pytorch import SparseConvTensor
import spconv.pytorch as spconv
from math import prod

DATAPATH = "../../datasets/DS"
# DATAPATH = "/work/riverlab/hongyu/dataset/DS"
DATALIST = "./filenames/DS_test.txt"

BATCH_SIZE = 1

c_u = 4.556890e+2
c_v = 1.976634e+2
f_u = 1.003556e+3
f_v = 1.003556e+3
baseline = 0.54

cudnn.benchmark = True

def Average(lst):
    return sum(lst) / len(lst)

def count_sparseConv(
    m,
    x,
    y
):
    x = x[0]
    kernel_size = tuple(m.kernel_size)
    os = (1,1,1)
    s = tuple(m.stride)
    d = tuple(m.dilation)
    ic, oc = m.in_channels, m.out_channels
    # if kernel_size != (1, 1, 1):
    #     kmaps = y.kmaps
    #     kmaps.update(x.kmaps)
    #     kmap_key = (os, kernel_size, s, d)
    #     kmap_size = kmaps[kmap_key][0].shape[0]
    #     total_ops = ic * oc * kmap_size
    # else:
    #     # FC
    #     total_ops = ic * oc * x.features.shape[0]

    total_ops = y.features.shape[0]*ic*oc

    m.total_ops += torch.DoubleTensor([int(total_ops)])


class WrappedModel(nn.Module):
	def __init__(self):
		super(WrappedModel, self).__init__()
		self.module = Voxel2D(192,"voxel") # that I actually define.
	def forward(self, *x):
		return self.module(*x)

if __name__ == '__main__':

    # load model
    model = WrappedModel()
    model = model.cuda()
    model.eval()
    ckpt_path = "/home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs/lr_0.001_batch_size_16_cost_vol_type_voxel_optimizer_adam_Voxel2D_sparse_weighted_loss_blind_/best.ckpt"
    # ckpt_path = "/home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs/lr_0.001_batch_size_16_cost_vol_type_voxel_optimizer_adam_Voxel2D_hie_weighted_loss_/best.ckpt"
    print("Loading model {}".format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['model'])

    # result list
    mac_list = []
    param_list = []

    test_dataset = VoxelDSDataset(DATAPATH, DATALIST, training=False, lite=False)
    TestImgLoader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    
    model.eval()
    
    total_count = len(TestImgLoader)*BATCH_SIZE
    invalid_count = 0

    t = tqdm(TestImgLoader)
    for batch_idx, sample in enumerate(t):
        left_img, right_img, disparity_batch, left_filename, vox_cost_vol_disps = sample['left'], sample['right'], sample['disparity'], sample['left_filename'], sample['vox_cost_vol_disps']
        voxel_grids = sample["voxel_grid"]

        input_L = left_img.cuda()
        input_R = right_img.cuda()
        try:
            macs, params = profile(model, inputs=(input_L, input_R, vox_cost_vol_disps), verbose=False,
                custom_ops={
                spconv.SparseConv3d: count_sparseConv,
                spconv.SparseConvTranspose3d: count_sparseConv,
                spconv.SubMConv3d: count_sparseConv
            })

            mac_list.append(macs)
            param_list.append(params)

            macs, params = clever_format([Average(mac_list), Average(param_list)], "%.3f")

            descrip_text = f"MACs: {macs}, Params: {params}"
            t.set_description(descrip_text)
            t.refresh()
            
        except:
            continue