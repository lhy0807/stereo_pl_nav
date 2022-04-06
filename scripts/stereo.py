from __future__ import print_function, division
import os
import torch.nn as nn
from mobilestereonet.utils import *
from mobilestereonet.utils.KittiColormap import *
import pandas as pd
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
from mobilestereonet.datasets.data_io import get_transform
import rospy
from sensor_msgs.msg import Image, CameraInfo
from mobilestereonet.models.MSNet2D import MSNet2D
from mobilestereonet.models.MSNet3D import MSNet3D
import logging
import coloredlogs
import time
import open3d as o3d

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)
CURR_DIR = os.path.dirname(__file__)

class Stereo():
    def listen_image(self, data: Image, side):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[24:,...]
        if side == "left":
            self.left_rect = frame
        elif side == "right":
            self.right_rect = frame
        else:
            raise NotImplementedError()

    def listen_camera_info(self, data: CameraInfo):
        self.right_camera_info = data

    def load_model(self):
        model = MSNet2D(192)
        model = nn.DataParallel(model)
        ckpt_path = os.path.join(CURR_DIR, "models/MSNet2D_SF_DS_KITTI2015.ckpt")
        logger.info("model {} loaded".format(ckpt_path))
        state_dict = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(state_dict['model'])
        self.model = model

    def calc_depth_map(self):
        if self.model is None or self.right_camera_info is None:
            return
        processed = get_transform()

        left_depth_rgb = self.left_rect[:, :, :3]
        depth_rgb = np.transpose(left_depth_rgb, (2, 0, 1))

        sample_left = processed(self.left_rect).numpy()
        sample_right = processed(self.right_rect).numpy()

        self.model.eval()

        sample_left = torch.Tensor(sample_left)
        sample_right = torch.Tensor(sample_right)

        sample_left = torch.unsqueeze(sample_left, dim=0)
        sample_right = torch.unsqueeze(sample_right, dim=0)

        c_u = self.right_camera_info.P[2]
        c_v = self.right_camera_info.P[6]
        f_u = self.right_camera_info.P[0]
        f_v = self.right_camera_info.P[5]
        b_x = self.right_camera_info.P[3] / (-f_u)  # relative
        b_y = self.right_camera_info.P[7] / (-f_v)

        def project_image_to_rect(uv_depth):
            ''' Input: nx3 first two channels are uv, 3rd channel
                    is depth in rect camera coord.
                Output: nx3 points in rect camera coord.
            '''
            n = uv_depth.shape[0]
            x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
            y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
            pts_3d_rect = np.zeros((n, 3))
            pts_3d_rect[:, 0] = x
            pts_3d_rect[:, 1] = y
            pts_3d_rect[:, 2] = uv_depth[:, 2]

            return pts_3d_rect

        def project_image_to_velo(uv_depth):
            pts_3d_rect = project_image_to_rect(uv_depth)
            return pts_3d_rect

        with torch.no_grad():
            disp_est_tn = self.model(sample_left, sample_right)[0]
            disp_est_np = tensor2numpy(disp_est_tn)
            disp_est = np.array(disp_est_np[0], dtype=np.float32)
            self.disp_image_pub.publish(
                self.bridge.cv2_to_imgmsg(kitti_colormap(disp_est), "bgr8"))

            disp_est[disp_est < 0] = 0
            baseline = -b_x/1e3
            mask = disp_est > 0
            depth = f_u * baseline / (disp_est + 1. - mask)
            self.depth_image_pub.publish(
                self.bridge.cv2_to_imgmsg(depth))

            # display point cloud
            rows, cols = depth.shape
            c, r = np.meshgrid(np.arange(cols), np.arange(rows))
            points = np.stack([c, r, depth])
            points = points.reshape((3, -1))
            points = points.T
            points = points[mask.reshape(-1)]
            cloud = project_image_to_velo(points)
            points_rgb = depth_rgb.reshape((3, -1)).T
            points_rgb = points_rgb.astype(float)
            points_rgb /= 255.

            rgbd_pcd = o3d.geometry.PointCloud()
            rgbd_pcd.points = o3d.utility.Vector3dVector(cloud)
            rgbd_pcd.colors = o3d.utility.Vector3dVector(points_rgb)
            
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.left_rect = None
        self.right_rect = None
        self.model = None
        self.right_camera_info = None

        rospy.Subscriber("/left_cam/image_rect_color", Image,
                         self.listen_image, "left", queue_size=1, buff_size=2**24)
        rospy.Subscriber("/right_cam/image_rect_color", Image,
                         self.listen_image, "right", queue_size=1, buff_size=2**24)
        rospy.Subscriber("/right_cam/camera_info", CameraInfo, self.listen_camera_info, queue_size=10)
        self.disp_image_pub = rospy.Publisher(
            "/disp_map", Image, queue_size=1)
        self.depth_image_pub = rospy.Publisher(
            "/depth_map", Image, queue_size=1)

        self.load_model()

if __name__ == "__main__":
    rospy.init_node("depth_map_gen")
    stereo = Stereo()

    while not rospy.is_shutdown():
        t1 = time.time()
        stereo.calc_depth_map()
        logger.info(f"Prediction used {round(time.time()-t1,2)}seconds")

    rospy.spin()