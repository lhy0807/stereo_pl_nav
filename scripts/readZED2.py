# import the opencv library
from tkinter import RIGHT
import cv2
import configparser
from matplotlib import image
import numpy as np
import os
import logging
import coloredlogs
import time
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

CURR_DIR = os.path.dirname(__file__)

LEFT_CAMERA_INFO = None
RIGHT_CAMERA_INFO = None

def read_calib(calib_file="SN28281527.conf"):
    config = configparser.ConfigParser()
    config.read(os.path.join(CURR_DIR, calib_file))

    baseline = float(config["STEREO"]["Baseline"])
    T = np.zeros((3,1), dtype=float)
    T[0,0] = baseline
    T[1,0] = float(config["STEREO"]["TY"])
    T[2,0] = float(config["STEREO"]["TZ"])

    left_cam_cx = float(config["LEFT_CAM_HD"]["cx"])
    left_cam_cy = float(config["LEFT_CAM_HD"]["cy"])
    left_cam_fx = float(config["LEFT_CAM_HD"]["fx"])
    left_cam_fy = float(config["LEFT_CAM_HD"]["fy"])
    left_cam_k1 = float(config["LEFT_CAM_HD"]["k1"])
    left_cam_k2 = float(config["LEFT_CAM_HD"]["k2"])
    left_cam_p1 = float(config["LEFT_CAM_HD"]["p1"])
    left_cam_p2 = float(config["LEFT_CAM_HD"]["p2"])
    left_cam_k3 = float(config["LEFT_CAM_HD"]["k3"])

    right_cam_cx = float(config["RIGHT_CAM_HD"]["cx"])
    right_cam_cy = float(config["RIGHT_CAM_HD"]["cy"])
    right_cam_fx = float(config["RIGHT_CAM_HD"]["fx"])
    right_cam_fy = float(config["RIGHT_CAM_HD"]["fy"])
    right_cam_k1 = float(config["RIGHT_CAM_HD"]["k1"])
    right_cam_k2 = float(config["RIGHT_CAM_HD"]["k2"])
    right_cam_p1 = float(config["RIGHT_CAM_HD"]["p1"])
    right_cam_p2 = float(config["RIGHT_CAM_HD"]["p2"])
    right_cam_k3 = float(config["RIGHT_CAM_HD"]["k3"])

    R_zed = np.zeros((1,3), dtype=float)
    R_zed[0,0] = float(config["STEREO"]["RX_HD"])
    R_zed[0,1] = float(config["STEREO"]["CV_HD"])
    R_zed[0,2] = float(config["STEREO"]["RZ_HD"])

    R,_ = cv2.Rodrigues(R_zed)

    cameraMatrix_left = np.zeros((3,3), dtype=float)
    cameraMatrix_left[0,0] = left_cam_fx
    cameraMatrix_left[0,2] = left_cam_cx
    cameraMatrix_left[1,1] = left_cam_fy
    cameraMatrix_left[1,2] = left_cam_cy
    cameraMatrix_left[2,2] = 1.0

    distCoeffs_left = np.zeros((5,1), dtype=float)
    distCoeffs_left[0,0] = left_cam_k1
    distCoeffs_left[1,0] = left_cam_k2
    distCoeffs_left[2,0] = left_cam_p1
    distCoeffs_left[3,0] = left_cam_p2
    distCoeffs_left[4,0] = left_cam_k3

    cameraMatrix_right = np.zeros((3,3), dtype=float)
    cameraMatrix_right[0,0] = right_cam_fx
    cameraMatrix_right[0,2] = right_cam_cx
    cameraMatrix_right[1,1] = right_cam_fy
    cameraMatrix_right[1,2] = right_cam_cy
    cameraMatrix_right[2,2] = 1.0

    distCoeffs_right = np.zeros((5,1), dtype=float)
    distCoeffs_right[0,0] = right_cam_k1
    distCoeffs_right[1,0] = right_cam_k2
    distCoeffs_right[2,0] = right_cam_p1
    distCoeffs_right[3,0] = right_cam_p2
    distCoeffs_right[4,0] = right_cam_k3
    
    image_size = (int(2560/2),720)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, image_size, R, T)

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, image_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, image_size, cv2.CV_32FC1)

    logger.info("Undistortion Rectify Map calculation finished.")
    logger.debug(f"Camera Matrix L: {cameraMatrix_left}")
    logger.debug(f"Camera Matrix R: {cameraMatrix_right}")

    left_cam_info = CameraInfo()
    left_cam_info.width = image_size[0]
    left_cam_info.height = image_size[1]
    left_cam_info.D = distCoeffs_left.T.tolist()[0]
    left_cam_info.K = cameraMatrix_left.reshape(-1,9).tolist()[0]
    left_cam_info.R = R1.reshape(-1,9).tolist()[0]
    left_cam_info.P = P1.reshape(-1,12).tolist()[0]
    left_cam_info.distortion_model = "plumb_bob"
    left_cam_info.header = Header()
    left_cam_info.header.stamp = rospy.Time.now()
    left_cam_info.header.frame_id = "zed2_left_frame"

    right_cam_info = CameraInfo()
    right_cam_info.width = image_size[0]
    right_cam_info.height = image_size[1]
    right_cam_info.D = distCoeffs_right.T.tolist()[0]
    right_cam_info.K = cameraMatrix_right.reshape(-1,9).tolist()[0]
    right_cam_info.R = R2.reshape(-1,9).tolist()[0]
    right_cam_info.P = P2.reshape(-1,12).tolist()[0]
    right_cam_info.distortion_model = "plumb_bob"
    right_cam_info.header = Header()
    right_cam_info.header.stamp = rospy.Time.now()
    right_cam_info.header.frame_id = "zed2_right_frame"

    global LEFT_CAMERA_INFO
    global RIGHT_CAMERA_INFO
    LEFT_CAMERA_INFO = left_cam_info
    RIGHT_CAMERA_INFO = right_cam_info

    return map_left_x, map_left_y, map_right_x, map_right_y

def main():

    rate = rospy.Rate(10)

    # read ZED2 factory calibration file
    map_left_x, map_left_y, map_right_x, map_right_y = read_calib()

    # define a video capture object
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    FRAME_WIDTH = int(2560/2)

    bridge = CvBridge()
    raw_left_image_pub = rospy.Publisher("/left_cam/image_color", Image, queue_size=10)
    raw_right_image_pub = rospy.Publisher("/right_cam/image_color", Image, queue_size=10)
    rect_left_image_pub = rospy.Publisher("/left_cam/image_rect_color", Image, queue_size=10)
    rect_right_image_pub = rospy.Publisher("/right_cam/image_rect_color", Image, queue_size=10)
    left_camear_info_pub = rospy.Publisher("/left_cam/camera_info", CameraInfo, queue_size=10)
    right_camear_info_pub = rospy.Publisher("/right_cam/camera_info", CameraInfo, queue_size=10)
    
    while not rospy.is_shutdown():    
        # Capture the video frame
        # by frame
        _, frame = vid.read()

        left_frame = frame[:,:FRAME_WIDTH]
        right_frame = frame[:,FRAME_WIDTH:]

        left_rect = cv2.remap(left_frame, map_left_x, map_left_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame, map_right_x, map_right_y, cv2.INTER_LINEAR)

        raw_left_image_pub.publish(bridge.cv2_to_imgmsg(left_frame,"bgr8"))
        rect_left_image_pub.publish(bridge.cv2_to_imgmsg(left_rect,"bgr8"))
        raw_right_image_pub.publish(bridge.cv2_to_imgmsg(right_frame,"bgr8"))
        rect_right_image_pub.publish(bridge.cv2_to_imgmsg(right_rect,"bgr8"))

        if LEFT_CAMERA_INFO is not None:
            LEFT_CAMERA_INFO.header.stamp = rospy.Time.now()
            left_camear_info_pub.publish(LEFT_CAMERA_INFO)

        if RIGHT_CAMERA_INFO is not None:
            RIGHT_CAMERA_INFO.header.stamp = rospy.Time.now()
            right_camear_info_pub.publish(RIGHT_CAMERA_INFO)

        rate.sleep()

    
    # After the loop release the cap object
    vid.release()

if __name__ == "__main__":
    try:
        rospy.init_node("readZED2")
        main()
    except rospy.ROSInterruptException:
        pass