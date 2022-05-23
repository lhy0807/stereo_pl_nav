from __future__ import print_function, division
import os
import torch.nn as nn
from voxelstereonet.models.mobilestereonet.utils import *
from voxelstereonet.models.mobilestereonet.utils.KittiColormap import *
import pandas as pd
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import logging
import coloredlogs
import time
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

coloredlogs.install(level="DEBUG")
CURR_DIR = os.path.dirname(__file__)

class Stereo():
    def listen_image(self, data: Image, side):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[:-95,:,:]
        if side == "left":
            self.left_rect = frame
        elif side == "right":
            self.right_rect = frame
        else:
            raise NotImplementedError()

    def listen_camera_info(self, data: CameraInfo):
        self.right_camera_info = data

    def calc_depth_map(self):
        if self.right_camera_info is None or self.left_rect is None or self.right_rect is None:
            rospy.logwarn("something is not ready")
            return

        c_u = self.right_camera_info.K[2]
        c_v = self.right_camera_info.K[5]
        f_u = self.right_camera_info.K[0]
        f_v = self.right_camera_info.K[4]
        b_x = self.right_camera_info.P[3] / (-self.right_camera_info.P[0])  # relative
        b_y = self.right_camera_info.P[7] / (-self.right_camera_info.P[5])

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

        disp_est = self.sgbm.compute(cv2.cvtColor(np.asarray(self.left_rect), cv2.COLOR_RGB2GRAY),
         cv2.cvtColor(np.asarray(self.right_rect), cv2.COLOR_RGB2GRAY))
        disp_est[disp_est < 0] = 0
        disp_est = disp_est/3040*192.
        self.disp_image_pub.publish(
            self.bridge.cv2_to_imgmsg(kitti_colormap(disp_est), "bgr8"))

        
        baseline = b_x
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
        # points = points[mask.reshape(-1)]
        cloud = project_image_to_velo(points)

        left_depth_rgb = self.left_rect[:, :, :3]
        depth_rgb = np.transpose(left_depth_rgb, (2, 0, 1))
        points_rgb = depth_rgb.reshape((3, -1)).T
        color_pl = points_rgb[:, 0] * 65536 + \
        points_rgb[:, 1] * 256 + points_rgb[:, 2]
        color_pl = np.expand_dims(color_pl, axis=-1)
        color_pl = color_pl.astype(np.uint32)

        # concat to ROS pointcloud foramt
        concat_pl = np.concatenate((cloud, color_pl), axis=1)
        points = concat_pl.tolist()

        # TODO: needs to fix this type conversion
        for i in range(len(points)):
            points[i][3] = int(points[i][3])

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        rgbd_pcd = o3d.geometry.PointCloud()
        rgbd_pcd.points = o3d.utility.Vector3dVector(cloud)
        rgbd_pcd.colors = o3d.utility.Vector3dVector(points_rgb)

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.camera_frame

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1),
                ]
        pc = pc2.create_cloud(header, fields, points)

        self.point_cloud_pub.publish(pc)
            
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.camera_frame = "zed_left"
        self.left_rect = None
        self.right_rect = None
        self.model = None
        self.right_camera_info = None

        rospy.Subscriber("/zed2/left/image_rect_color", Image,
                         self.listen_image, "left", queue_size=1, buff_size=2**24)
        rospy.Subscriber("/zed2/right/image_rect_color", Image,
                         self.listen_image, "right", queue_size=1, buff_size=2**24)
        rospy.Subscriber("/zed2/right/camera_info", CameraInfo, self.listen_camera_info, queue_size=10)
        self.disp_image_pub = rospy.Publisher(
            "/disp_map", Image, queue_size=1)
        self.depth_image_pub = rospy.Publisher(
            "/depth_map", Image, queue_size=1)
        self.point_cloud_pub = rospy.Publisher(
            "/pointcloud", PointCloud2, queue_size=1)

        #Note: disparity range is tuned according to specific parameters obtained through trial and error. 
        win_size = 5
        min_disp = -1
        max_disp = 191 #min_disp * 9
        num_disp = max_disp - min_disp # Needs to be divisible by 16
        #Create Block matching object. 
        self.sgbm = cv2.StereoSGBM_create(minDisparity= min_disp,
        numDisparities = num_disp,
        blockSize = 5,
        uniquenessRatio = 5,
        speckleWindowSize = 5,
        speckleRange = 5,
        disp12MaxDiff = 1,
        P1 = 8*3*win_size**2,#8*3*win_size**2,
        P2 =32*3*win_size**2) #32*3*win_size**2)

if __name__ == "__main__":
    rospy.loginfo("Stereo Node started!")
    rospy.init_node("depth_map_gen")
    stereo = Stereo()

    while not rospy.is_shutdown():
        t1 = time.time()
        stereo.calc_depth_map()
        rospy.loginfo(f"Prediction used {round(time.time()-t1,2)}seconds")

    rospy.spin()