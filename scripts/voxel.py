from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
from cv_bridge import CvBridge
import cv2
from voxelstereonet.models.mobilestereonet.datasets.data_io import get_transform
import rospy
from voxelstereonet.models.Voxel2D import Voxel2D
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import logging
import coloredlogs
import time
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

coloredlogs.install(level="DEBUG")
CURR_DIR = os.path.dirname(__file__)

class Stereo():
    def calc_disp(self):

        f_u = 1.003556e+3
        baseline = 0.54

        # calculate voxel cost volume disparity set
        vox_cost_vol_disp_set = set()
        max_disp = 192
        # depth starting from voxel_size since 0 will cause issue
        for z in np.arange(0.5, 32, 0.5*3):
            # get respective disparity
            d = f_u * baseline / z

            if d > max_disp:
                continue

            # real disparity -> disparity in feature map
            vox_cost_vol_disp_set.add(round(d/4))

        vox_cost_vol_disps = list(vox_cost_vol_disp_set)
        vox_cost_vol_disps = sorted(vox_cost_vol_disps)

        tmp = []
        for i in vox_cost_vol_disps:
            tmp.append(torch.unsqueeze(torch.Tensor([i]), 0))
        self.vox_cost_vol_disps = tmp
        rospy.loginfo("Disparity level calculated")
        
    def listen_image(self, data: Image, side):
        frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[160:-160,200:-200,:]
        if side == "left":
            self.left_rect = frame
        elif side == "right":
            self.right_rect = frame
        else:
            raise NotImplementedError()

    def listen_camera_info(self, data: CameraInfo):
        self.right_camera_info = data

    def load_model(self):
        rospy.loginfo("start loading model")
        voxel_model = Voxel2D(192, "voxel")
        voxel_model = nn.DataParallel(voxel_model)
        voxel_model.cuda()
        ckpt_path = os.path.join(CURR_DIR, "voxelstereonet/logs/lr_0.001_batch_size_16_cost_vol_type_voxel_optimizer_adam_/best.ckpt")
        rospy.loginfo("model {} loaded".format(ckpt_path))
        state_dict = torch.load(ckpt_path, map_location="cuda")
        voxel_model.load_state_dict(state_dict['model'])
        rospy.loginfo("model weight loaded")
        self.model = voxel_model

    def calc_depth_map(self):
        if self.model is None or self.right_camera_info is None or self.left_rect is None or self.right_rect is None:
            rospy.logwarn("something is not ready")
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

        with torch.no_grad():
            vox_pred = self.model(sample_left, sample_right, self.vox_cost_vol_disps)[0][0]
            
            vox_pred = vox_pred.detach().cpu().numpy()
            vox_pred[vox_pred < 0.5] = 0
            vox_pred[vox_pred >= 0.5] = 1
            
            offsets = np.array([32, 62, 0])
            voxel_size = 0.5
            xyz_pred = np.asarray(np.where(vox_pred == 1)) # get back indexes of populated voxels
            cloud_pred = np.asarray([(pt-offsets)*voxel_size for pt in xyz_pred.T])

            rospy.logdebug(f"Size of point cloud: {len(cloud_pred)}")
            new_pl = np.zeros((len(cloud_pred), 3))
            new_pl[:, 0] = cloud_pred[:, 2]
            new_pl[:, 1] = -cloud_pred[:, 0]
            new_pl[:, 2] = -cloud_pred[:, 1]
            cloud = new_pl

            points_rgb = np.ones((len(cloud_pred), 1))
            color_pl = points_rgb[:, 0] * 65536 * 255
            color_pl = np.expand_dims(color_pl, axis=-1)
            color_pl = color_pl.astype(np.uint32)

            # concat to ROS pointcloud foramt
            concat_pl = np.concatenate((cloud, color_pl), axis=1)
            points = concat_pl.tolist()

            # TODO: needs to fix this type conversion
            for i in range(len(points)):
                points[i][3] = int(points[i][3])

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
        self.camera_frame = "camera_rgb_frame"
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
        self.point_cloud_pub = rospy.Publisher(
            "/pointcloud", PointCloud2, queue_size=1)

        self.load_model()

        self.vox_cost_vol_disps = []
        self.calc_disp()

if __name__ == "__main__":
    rospy.loginfo("Stereo Node started!")
    rospy.init_node("depth_map_gen", log_level=rospy.DEBUG)
    stereo = Stereo()

    while not rospy.is_shutdown():
        t1 = time.time()
        stereo.calc_depth_map()
        rospy.logdebug(f"Prediction used {round(time.time()-t1,2)}seconds")

    rospy.spin()