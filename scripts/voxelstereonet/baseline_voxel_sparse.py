from __future__ import print_function, division
import enum
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import *
from utils.KittiColormap import *
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from PIL import Image
from tqdm import tqdm, trange
from models.Voxel2D_sparse import Voxel2D
from sklearn.metrics import accuracy_score, f1_score
import coloredlogs, logging
from datasets import VoxelDSDataset
import torch.nn.functional as F
import traceback
from torchmetrics.functional import jaccard_index
from pytorch3d.loss import chamfer_distance

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

DATAPATH = "../../datasets/DS"
# DATAPATH = "/work/riverlab/hongyu/dataset/DS"
DATALIST = "./filenames/DS_test.txt"

BATCH_SIZE = 16

c_u = 4.556890e+2
c_v = 1.976634e+2
f_u = 1.003556e+3
f_v = 1.003556e+3
baseline = 0.54

cudnn.benchmark = True

def Average(lst):
    return sum(lst) / len(lst)

if __name__ == '__main__':

    # load model
    model = Voxel2D(192, "voxel")
    model = nn.DataParallel(model)
    model.cuda()
    ckpt_path = "/home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs/lr_0.001_batch_size_16_cost_vol_type_voxel_optimizer_adam_Voxel2D_sparse_weighted_loss_/best.ckpt"
    print("Loading model {}".format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['model'])

    # result list
    cd_list = {0:[],1:[],2:[],3:[]}
    iou_list = {0:[],1:[],2:[],3:[]}

    test_dataset = VoxelDSDataset(DATAPATH, DATALIST, training=False, lite=False)
    TestImgLoader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)
    
    model.eval()
    
    total_count = len(TestImgLoader)*BATCH_SIZE
    invalid_count = 0

    t = tqdm(TestImgLoader)
    for batch_idx, sample in enumerate(t):
        left_img, right_img, disparity_batch, left_filename, vox_cost_vol_disps = sample['left'], sample['right'], sample['disparity'], sample['left_filename'], sample['vox_cost_vol_disps']
        voxel_grids = sample["voxel_grid"]

        # predict disparity map
        with torch.no_grad():
            disp_est_tn = model(left_img.cuda(), right_img.cuda(), vox_cost_vol_disps)[0]
        try:
            for level, level_pred in enumerate(disp_est_tn):
                disp_est_np = tensor2numpy(level_pred)
                for idx, disp_est in enumerate(disp_est_np):
                    vox_grid_gt  = voxel_grids[level][idx]
                    vox_grid_gt = tensor2numpy(vox_grid_gt)
                    
                    voxel_size = 32/vox_grid_gt.shape[0]

                    vox_pred = disp_est
                    vox_pred[vox_pred < 0.5] = 0
                    vox_pred[vox_pred >= 0.5] = 1
                    offsets = np.array([32, 62, 0])
                    xyz_pred = np.asarray(np.where(vox_pred == 1)) # get back indexes of populated voxels
                    cloud_pred = np.asarray([(pt-offsets)*voxel_size for pt in xyz_pred.T])

                    xyz_v = np.asarray(np.where(vox_grid_gt == 1)) # get back indexes of populated voxels
                    cloud_np_gt = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])

                    intersect = vox_pred*vox_grid_gt  # Logical AND
                    union = vox_pred+vox_grid_gt  # Logical OR

                    IoU = ((intersect.sum() + 1.0) / (union.sum() - intersect.sum() + 1.0))
                    iou_list[level].append(IoU)

                    cd = chamfer_distance(torch.Tensor(np.expand_dims(cloud_pred,0)), torch.Tensor(np.expand_dims(cloud_np_gt,0)))[0]
                    cd_list[level].append(cd)

            descrip_text = ""
            for i in range(4):
                descrip_text += f"Level {i} CD is {Average(cd_list[i]).type(torch.float16)}, Level {i} IoU is {Average(iou_list[i]).astype(np.float16)}"
            t.set_description(descrip_text)
            t.refresh()
        except Exception as e:
            logger.warning(f"Something bad happended {traceback.format_exc()}")
            logger.warning(f"cloud_pred shape {cloud_pred.shape}, cloud_np_gt shape {cloud_np_gt.shape}")