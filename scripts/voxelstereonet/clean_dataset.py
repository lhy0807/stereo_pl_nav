import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from tqdm import trange
from tqdm.contrib.concurrent import process_map

c_u = 479.5
c_v = 269.5
f_u = 1050.0
f_v = 1050.0
baseline = 0.1
voxel_size = 0.05
datapath = "/home/chris/pl_ws/src/stereo_pl_nav/datasets/SceneFlow"

def load_path(list_filename):
    lines = read_all_lines(list_filename)
    splits = [line.split() for line in lines]
    left_images = []
    right_images = []
    disp_images = []
    for x in splits:
        if "15mm_focallength" in x[0]:
            continue
        left_images.append(x[0])
        right_images.append(x[1])
        disp_images.append(x[2])
    return left_images, right_images, disp_images

def load_image(filename):
        return Image.open(filename).convert('RGB')

def load_disp(filename):
    data, scale = pfm_imread(filename)
    data = np.ascontiguousarray(data, dtype=np.float32)
    return data
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

left_filenames, right_filenames, disp_filenames = load_path("./filenames/sceneflow_train.txt")

def single_process(index):
    left_img = load_image(os.path.join(datapath, left_filenames[index]))
    right_img = load_image(os.path.join(datapath, right_filenames[index]))
    disparity = load_disp(os.path.join(datapath, disp_filenames[index]))
    w, h = left_img.size
    crop_w, crop_h = 960, 512

    left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
    right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
    disparity = disparity[h - crop_h:h, w - crop_w: w]

    processed = get_transform()
    left_img = processed(left_img)
    right_img = processed(right_img)

    # calcualte depth for ground truth disparity map
    mask = disparity > 0
    depth_gt = f_u * baseline / (disparity + 1. - mask)
    vox_grid_gt = torch.zeros((int(2.4/voxel_size)+1, int(1.2/voxel_size)+1, int(2.4/voxel_size)+1))
    try:
        cloud_gt = calc_cloud(disparity, depth_gt)
        filtered_cloud_gt = filter_cloud(cloud_gt)
        vox_grid_gt,cloud_np_gt  = calc_voxel_grid(filtered_cloud_gt, voxel_size=voxel_size)

        if np.count_nonzero(vox_grid_gt) <= 4:
            print(f"{left_filenames[index]} too short")
    except Exception as e:
        print("error here")
        pass

def main():
    r = process_map(single_process, range(len(left_filenames)), max_workers=16, chunksize=1)     

main()