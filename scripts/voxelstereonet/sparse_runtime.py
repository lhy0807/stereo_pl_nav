from tkinter.tix import Tree
from thop import profile, clever_format
from ptflops import get_model_complexity_info
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from zmq import device
# from models.Voxel2D_sparse import Voxel2D
from models.Voxel2D_hie import Voxel2D

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from datasets import VoxelDSDataset
from tqdm import tqdm
import time
import torch.autograd.profiler as profiler
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATAPATH = "../../datasets/DS"
# DATAPATH = "/work/riverlab/hongyu/dataset/DS"
DATALIST = "./filenames/DS_test.txt"

BATCH_SIZE = 1

c_u = 4.556890e+2
c_v = 1.976634e+2
f_u = 1.003556e+3
f_v = 1.003556e+3
baseline = 0.54

use_cuda = True if torch.cuda.is_available() else False

if use_cuda:
    cudnn.benchmark = True

def Average(lst):
    return sum(lst) / len(lst)


class WrappedModel(nn.Module):
	def __init__(self):
		super(WrappedModel, self).__init__()
		self.module = Voxel2D(192,"voxel") # that I actually define.
	def forward(self, *x):
		return self.module(*x)

if __name__ == '__main__':

    # load model
    model = WrappedModel()
    if use_cuda:
        model = model.cuda()
    model.eval()
    # ckpt_path = "/home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs/lr_0.001_batch_size_16_cost_vol_type_voxel_optimizer_adam_Voxel2D_sparse_weighted_loss_blind_/best.ckpt"
    ckpt_path = "/home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs/lr_0.001_batch_size_16_cost_vol_type_voxel_optimizer_adam_Voxel2D_hie_weighted_loss_/best.ckpt"
    print("Loading model {}".format(ckpt_path))
    if use_cuda:
        state_dict = torch.load(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, "cpu")
    model.load_state_dict(state_dict['model'])

    test_dataset = VoxelDSDataset(DATAPATH, DATALIST, training=False, lite=False)
    TestImgLoader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    
    model.eval()
    
    total_count = len(TestImgLoader)*BATCH_SIZE
    invalid_count = 0
    time_list = []

    if use_cuda:
        torch.cuda.synchronize()
    t = tqdm(TestImgLoader)
    for batch_idx, sample in enumerate(t):
        left_img, right_img, disparity_batch, left_filename, vox_cost_vol_disps = sample['left'], sample['right'], sample['disparity'], sample['left_filename'], sample['vox_cost_vol_disps']
        voxel_grids = sample["voxel_grid"]

        if use_cuda:
            left_img = left_img.cuda()
            right_img = right_img.cuda()
        model(left_img, right_img, vox_cost_vol_disps)
        t1 = time.time()
        try:
            # with profiler.profile(with_stack=True, profile_memory=True, use_cuda=True, with_flops=True) as prof:
            model(left_img, right_img, vox_cost_vol_disps)
            if use_cuda:
                torch.cuda.synchronize()
            time_spent = time.time()-t1
            # time_list.append(time_spent)
            print(time_spent)
            # print(torch.cuda.mem_get_info(0))
            # print(prof.key_averages(group_by_stack_n=20).table(sort_by='cuda_memory_usage', row_limit=20))
        except:
            pass

        gc.collect()