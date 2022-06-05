from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
from cv_bridge import CvBridge
import cv2
from voxelstereonet.models.mobilestereonet.datasets.data_io import get_transform
import rospy
from voxelstereonet.models.Voxel2D_lite import Voxel2D
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import logging
import coloredlogs
import time
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

coloredlogs.install(level="DEBUG")
torch.backends.cudnn.benchmark = True
CURR_DIR = os.path.dirname(__file__)

class Stereo():
    def calc_disp(self):

        f_u = 365.68
        baseline = 0.12

        # calculate voxel cost volume disparity set
        vox_cost_vol_disp_set = set()
        max_disp = 192
        # depth starting from voxel_size since 0 will cause issue
        for z in np.arange(0.1, 6.4, 0.05):
            # get respective disparity
            d = f_u * baseline / z

            if d > max_disp:
                continue

            # real disparity -> disparity in feature map
            vox_cost_vol_disp_set.add(round(d/8))

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
        self.image_timestamp = data.header.stamp

    def listen_camera_info(self, data: CameraInfo):
        self.right_camera_info = data

    def load_model(self):
        rospy.loginfo("start loading model")
        voxel_model = Voxel2D(192, "voxel")
        voxel_model = nn.DataParallel(voxel_model)
        if torch.cuda.is_available():
            voxel_model.cuda()
        ckpt_path = os.path.join(CURR_DIR, "voxelstereonet/logs/lr_0.001_batch_size_32_cost_vol_type_voxel_optimizer_adam_finetune/best.ckpt")
        rospy.loginfo("model {} loaded".format(ckpt_path))
        if torch.cuda.is_available():
            state_dict = torch.load(ckpt_path, map_location="cuda")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        voxel_model.load_state_dict(state_dict['model'])
        rospy.loginfo("model weight loaded")
        self.model = voxel_model
        self.model.eval()

    def calc_depth_map(self):
        if self.model is None:
            rospy.logwarn("model is not ready")
            return
        if self.right_camera_info is None:
            rospy.logwarn("camera info is not ready")
            return
        if self.left_rect is None:
            rospy.logwarn("Left Rect Image is not ready")
            return
        if self.right_rect is None:
            rospy.logwarn("Right Rect Image is not ready")
            return

        init_time = rospy.Time.now()

        processed = get_transform()

        sample_left = processed(self.left_rect)
        sample_right = processed(self.right_rect)

        sample_left = torch.unsqueeze(sample_left, dim=0)
        sample_right = torch.unsqueeze(sample_right, dim=0)

        c_u = self.right_camera_info.P[2]
        c_v = self.right_camera_info.P[6]
        f_u = self.right_camera_info.P[0]
        f_v = self.right_camera_info.P[5]

        with torch.no_grad():
            vox_pred = self.model(sample_left, sample_right, self.vox_cost_vol_disps)[0][0]
            
            vox_pred = vox_pred.detach().cpu().numpy()
            vox_pred[vox_pred < 0.5] = 0
            vox_pred[vox_pred >= 0.5] = 1
            
            offsets = np.array([32, 63, 0])
            voxel_size = 0.1
            xyz_pred = np.asarray(np.where(vox_pred == 1)) # get back indexes of populated voxels
            cloud = np.asarray([(pt-offsets)*voxel_size for pt in xyz_pred.T])

            rospy.logdebug(f"Size of point cloud: {len(cloud)}")


            points = cloud.tolist()
            header = Header()
            # header.stamp = init_time
            header.stamp = self.image_timestamp
            header.frame_id = self.camera_frame

            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  
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

        self.image_timestamp = None

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