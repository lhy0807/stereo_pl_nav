from __future__ import print_function, division
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
from models.coex.stereo import Stereo
import coloredlogs, logging
from datasets import VoxelDSDataset
import torch.nn.functional as F
import traceback
from ruamel.yaml import YAML
from pytorch3d.loss import chamfer_distance

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

DATAPATH = "../../datasets/DS"
# DATAPATH = "/work/riverlab/hongyu/dataset/DS"
DATALIST = "./filenames/DS_test.txt"

VOXEL_SIZE = 0.5
BATCH_SIZE = 2

c_u = 4.556890e+2
c_v = 1.976634e+2
f_u = 1.003556e+3
f_v = 1.003556e+3
baseline = 0.54

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
    min_mask = cloud >= [-16,-31,0.0]
    max_mask = cloud <= [16,1,32]
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask
    filtered_cloud = cloud[filter_mask]
    return filtered_cloud

def calc_voxel_grid(filtered_cloud, voxel_size):
    xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int) # quantized point values, here you will loose precision
    vox_grid = np.zeros((int(32/voxel_size), int(32/voxel_size), int(32/voxel_size))) #Empty voxel grid
    offsets = np.array([32, 62, 0])
    xyz_offset_q = xyz_q+offsets
    vox_grid[xyz_offset_q[:,0],xyz_offset_q[:,1],xyz_offset_q[:,2]] = 1 # Setting all voxels containitn a points equal to 1

    xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels
    cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])
    return vox_grid, cloud_np

def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    backbone_cfg = YAML().load(
        open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg

if __name__ == '__main__':

    config = 'cfg_coex.yaml'
    version = 0  # CoEx

    cfg = load_configs(
        './models/coex/configs/stereo/{}'.format(config))

    ckpt = '{}/{}/version_{}/checkpoints/last.ckpt'.format(
        './models/coex/logs/stereo', cfg['model']['name'], version)
    cfg['stereo_ckpt'] = ckpt
    # load model
    model = Stereo.load_from_checkpoint(cfg['stereo_ckpt'],
                                                strict=False,
                                                cfg=cfg).cuda()

    # result list
    loss_list = []
    cd_list = []
    iou_list = []

    test_dataset = VoxelDSDataset(DATAPATH, DATALIST, training=False)
    TestImgLoader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False)
    
    model.eval()
    
    total_count = len(TestImgLoader)*BATCH_SIZE
    invalid_count = 0

    t = tqdm(TestImgLoader)
    for batch_idx, sample in enumerate(t):
        left_img, right_img, disparity_batch, left_filename = sample['left'], sample['right'], sample['disparity'], sample['left_filename']
        voxel_grids = sample["voxel_grid"]
        voxel_grids = voxel_grids.cpu().numpy()
        # disparity_batch = disparity_batch.cpu().numpy()


        # predict disparity map
        with torch.no_grad():
            disp_est_tn = model(left_img.cuda(), right_img.cuda())[0]
            disp_est_np = tensor2numpy(disp_est_tn)
            disp_est_np = np.expand_dims(disp_est_np, axis=0)
        try:
            for idx, disp_est in enumerate(disp_est_np):
                vox_grid_gt  = voxel_grids[idx]
                if vox_grid_gt.max() == 0.0:
                    invalid_count += 1
                    continue

                disp_est = np.array(disp_est, dtype=np.float32)
                disp_est[disp_est < 0] = 0

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

                cloud_np_gt = np.asarray(np.where(vox_grid_gt == 1))
                if cloud_np.shape[0] < 32 or cloud_np_gt.shape[1] < 32:
                    invalid_count += 1

                offsets = np.array([32, 62, 0])
                xyz_v = np.asarray(np.where(vox_grid_gt == 1)) # get back indexes of populated voxels
                cloud_np_gt = np.asarray([(pt-offsets)*VOXEL_SIZE for pt in xyz_v.T])

                intersect = vox_grid*vox_grid_gt  # Logical AND
                union = vox_grid+vox_grid_gt  # Logical OR

                IoU = ((intersect.sum() + 1.0) / (union.sum() - intersect.sum() + 1.0))
                iou_list.append(IoU)

                cd = chamfer_distance(torch.Tensor(np.expand_dims(cloud_np,0)), torch.Tensor(np.expand_dims(cloud_np_gt,0)))[0]
                cd_list.append(cd)

                t.set_description(f"CD is {Average(cd_list)}, IoU is {Average(iou_list)}, Invalid Sample {invalid_count} out of {total_count} @ {round(invalid_count/total_count*100, 2)}%")
                t.refresh()
        except Exception as e:
            logger.warning(f"Something bad happended {traceback.format_exc()}")