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
        frame = frame[:-95,:,:]
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
        if torch.cuda.is_available():
            voxel_model.cuda()
        ckpt_path = os.path.join(CURR_DIR, "voxelstereonet/logs/lr_0.001_batch_size_16_cost_vol_type_voxel_optimizer_adam_/best.ckpt")
        rospy.loginfo("model {} loaded".format(ckpt_path))
        if torch.cuda.is_available():
            state_dict = torch.load(ckpt_path, map_location="cuda")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        voxel_model.load_state_dict(state_dict['model'])
        rospy.loginfo("model weight loaded")
        self.model = voxel_model

    def calc_depth_map(self):
        if self.model is None or self.right_camera_info is None or self.left_rect is None or self.right_rect is None:
            rospy.logwarn("something is not ready")
            return

        init_time = rospy.Time.now()
        self.model.eval()

        processed = get_transform()

        sample_left = processed(self.left_rect)
        sample_right = processed(self.right_rect)

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
            cloud = np.asarray([(pt-offsets)*voxel_size for pt in xyz_pred.T])

            rospy.logdebug(f"Size of point cloud: {len(cloud)}")

            # convert KITTI to ZED2
            K_c_u = 4.556890e+2
            K_c_v = 1.976634e+2
            K_f_u = 1.003556e+3
            K_f_v = 1.003556e+3
            K_baseline = 0.54
            # step1: calculate u,v 
            uv_depth = np.zeros((len(cloud), 2))
            for i in range(len(cloud)):
                uv_depth[i,0] = (cloud[i,0]*K_f_u)/cloud[i,2] + K_c_u
                uv_depth[i,1] = (cloud[i,1]*K_f_v)/cloud[i,2] + K_c_v

            # step2: generate new cloud using ZED2
            try:
                new_cloud = np.zeros(cloud.shape)
                new_cloud[:,2] = cloud[:,2] / (1.003556e+3*0.54 / 532.86 / 0.12)
                new_cloud[:,0] = ((uv_depth[:, 0] - c_u) * new_cloud[:, 2]) / f_u
                new_cloud[:,1] = ((uv_depth[:, 1] - c_v) * new_cloud[:, 2]) / f_v
            except Exception:
                rospy.logwarn("No PointCloud detected!")
                return
            cloud = new_cloud

            points_rgb = np.ones((len(cloud), 1))
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
            header.stamp = init_time
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
        self.point_cloud_pub = rospy.Publisher(
            "/voxels", PointCloud2, queue_size=1)

        self.load_model()

        self.vox_cost_vol_disps = []
        self.calc_disp()

if __name__ == "__main__":
    rospy.loginfo("Stereo Node started!")
    rospy.init_node("voxel_gen", log_level=rospy.DEBUG)
    stereo = Stereo()

    while not rospy.is_shutdown():
        t1 = time.time()
        stereo.calc_depth_map()
        rospy.loginfo(f"Prediction used {round(time.time()-t1,2)}seconds")

    rospy.spin()