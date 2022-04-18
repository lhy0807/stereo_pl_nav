from __future__ import print_function, division
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import *
from utils.KittiColormap import *
from pyntcloud import PyntCloud
import pandas as pd
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from PIL import Image
from tqdm import tqdm, trange
from models.MSNet2D import MSNet2D
from models.MSNet3D import MSNet3D
from models.Voxel2D import Voxel2D
from sklearn.metrics import accuracy_score, f1_score
import coloredlogs, logging
from datasets import VoxelDataset, SceneFlowDataset
import torch.nn.functional as F
import traceback
from torchmetrics.functional import jaccard_index

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

DATAPATH = "../../datasets/SceneFlow"
DATALIST = "./filenames/sceneflow_test.txt"

VOXEL_SIZE = 0.05 

cudnn.benchmark = True

def Average(lst):
    return sum(lst) / len(lst)

def calc_cloud(disp_est, depth):
    mask = disp_est > 0
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = project_image_to_velo(points)
    return cloud

def filter_cloud(cloud):
    min_mask = cloud >= [-1.2,-1.0,0.0]
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = cloud <= [1.2,0.2,2.4]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask
    filtered_cloud = cloud[filter_mask]
    return filtered_cloud

def calc_voxel_grid(filtered_cloud, voxel_size):
    xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int) # quantized point values, here you will loose precision
    vox_grid = np.zeros((int(2.4/voxel_size)+1, int(1.2/voxel_size)+1, int(2.4/voxel_size)+1)) #Empty voxel grid
    offsets = np.array([24, 20, 0])
    xyz_offset_q = xyz_q+offsets
    vox_grid[xyz_offset_q[:,0],xyz_offset_q[:,1],xyz_offset_q[:,2]] = 1 # Setting all voxels containitn a points equal to 1

    xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels
    cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])
    return vox_grid, cloud_np

if __name__ == '__main__':

    # load model
    model = MSNet2D(192)
    model = nn.DataParallel(model)
    model.cuda()
    ckpt_path = "../models/MSNet2D_SF_DS_KITTI2015.ckpt"
    print("Loading model {}".format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict['model'])

    # result list
    loss_list = []
    acc_list = []
    iou_list = []

    test_dataset = VoxelDataset(DATAPATH, DATALIST, training=False)
    TestImgLoader = DataLoader(test_dataset, 3, shuffle=True, num_workers=6, drop_last=False)
    
    model.eval()
    
    t = tqdm(TestImgLoader)
    for sample in t:
        left_img, right_img, disparity_batch, left_filename = sample['left'], sample['right'], sample['disparity'], sample['left_filename']
        voxel_grid = sample["voxel_grid"]
        voxel_grid = voxel_grid.cpu().numpy()
        # disparity_batch = disparity_batch.cpu().numpy()
        # Camera intrinsics
        # 15mm images have different focals
        c_u = 479.5
        c_v = 269.5
        f_u = 1050.0
        f_v = 1050.0
        baseline = 0.1

        # predict disparity map
        with torch.no_grad():
            disp_est_tn = model(left_img.cuda(), right_img.cuda())[0]
            disp_est_np = tensor2numpy(disp_est_tn)
        try:
            for idx, disp_est in enumerate(disp_est_np):
                vox_grid_gt  = voxel_grid[idx]
                if vox_grid_gt.max() == 0.0:
                    continue

                disp_est = np.array(disp_est, dtype=np.float32)
                disp_est[disp_est < 0] = 0

                if "15mm_focallength" in left_filename[idx]:
                    f_u = 450.0
                    f_v = 450.0

                # calcualte depth for predicted disparity map
                mask = disp_est > 0
                depth = f_u * baseline / (disp_est + 1. - mask)


                def project_image_to_rect(uv_depth):
                    ''' Input: nx3 first two channels are uv, 3rd channel
                            is depth in rect camera coord.
                        Output: nx3 points in rect camera coord.
                    '''
                    n = uv_depth.shape[0]
                    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + baseline
                    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v
                    pts_3d_rect = np.zeros((n, 3))
                    pts_3d_rect[:, 0] = x
                    pts_3d_rect[:, 1] = y
                    pts_3d_rect[:, 2] = uv_depth[:, 2]
                    return pts_3d_rect

                def project_image_to_velo(uv_depth):
                    pts_3d_rect = project_image_to_rect(uv_depth)
                    return pts_3d_rect

                # calculate point cloud for estimation
                cloud = calc_cloud(disp_est, depth)

                # crop our point cloud ROI
                filtered_cloud = filter_cloud(cloud)

                vox_grid,cloud_np  = calc_voxel_grid(filtered_cloud, voxel_size=VOXEL_SIZE)

                loss = F.binary_cross_entropy(torch.from_numpy(vox_grid), torch.from_numpy(vox_grid_gt))
                iou = jaccard_index(torch.from_numpy(vox_grid).type(torch.IntTensor), torch.from_numpy(vox_grid_gt).type(torch.IntTensor), num_classes=2)
                
                loss_list.append(loss)
                iou_list.append(iou)

                t.set_description(f"Loss is {Average(loss_list)}, IoU is {Average(iou_list)}")
                t.refresh()
        except Exception as e:
            logger.warning(f"Something bad happended {traceback.format_exc()}")