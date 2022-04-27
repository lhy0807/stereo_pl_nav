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

    left_cam_cx = float(config["LEFT_CAM_VGA"]["cx"])
    left_cam_cy = float(config["LEFT_CAM_VGA"]["cy"])
    left_cam_fx = float(config["LEFT_CAM_VGA"]["fx"])
    left_cam_fy = float(config["LEFT_CAM_VGA"]["fy"])
    left_cam_k1 = float(config["LEFT_CAM_VGA"]["k1"])
    left_cam_k2 = float(config["LEFT_CAM_VGA"]["k2"])
    left_cam_p1 = float(config["LEFT_CAM_VGA"]["p1"])
    left_cam_p2 = float(config["LEFT_CAM_VGA"]["p2"])
    left_cam_k3 = float(config["LEFT_CAM_VGA"]["k3"])

    right_cam_cx = float(config["RIGHT_CAM_VGA"]["cx"])
    right_cam_cy = float(config["RIGHT_CAM_VGA"]["cy"])
    right_cam_fx = float(config["RIGHT_CAM_VGA"]["fx"])
    right_cam_fy = float(config["RIGHT_CAM_VGA"]["fy"])
    right_cam_k1 = float(config["RIGHT_CAM_VGA"]["k1"])
    right_cam_k2 = float(config["RIGHT_CAM_VGA"]["k2"])
    right_cam_p1 = float(config["RIGHT_CAM_VGA"]["p1"])
    right_cam_p2 = float(config["RIGHT_CAM_VGA"]["p2"])
    right_cam_k3 = float(config["RIGHT_CAM_VGA"]["k3"])

    R_zed = np.zeros((1,3), dtype=float)
    R_zed[0,0] = float(config["STEREO"]["RX_VGA"])
    R_zed[0,1] = float(config["STEREO"]["CV_VGA"])
    R_zed[0,2] = float(config["STEREO"]["RZ_VGA"])

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
    
    image_size = (int(1344/2),376)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, image_size, R, T)

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, image_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, image_size, cv2.CV_32FC1)

    logger.info("Undistortion Rectify Map calculation finished.")
    logger.debug(f"Camera Matrix L: {cameraMatrix_left}")
    logger.debug(f"Camera Matrix R: {cameraMatrix_right}")

    return map_left_x, map_left_y, map_right_x, map_right_y

def main():

    rate = rospy.Rate(1)

    # read ZED2 factory calibration file
    map_left_x, map_left_y, map_right_x, map_right_y = read_calib()

    # define a video capture object
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
    FRAME_WIDTH = int(1344/2)
    
    while not rospy.is_shutdown():    
        # Capture the video frame
        # by frame
        frame_time = int(time.time()*10)
        _, frame = vid.read()

        left_frame = frame[:,:FRAME_WIDTH]
        right_frame = frame[:,FRAME_WIDTH:]

        left_rect = cv2.remap(left_frame, map_left_x, map_left_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_frame, map_right_x, map_right_y, cv2.INTER_LINEAR)

        cv2.imwrite(f"left/{frame_time}.jpg", left_rect)
        cv2.imwrite(f"right/{frame_time}.jpg", right_rect)
        rate.sleep()

    
    # After the loop release the cap object
    vid.release()

if __name__ == "__main__":
    main()