import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import tf
import open3d as o3d
from std_msgs.msg import Header

topics = ["/zed2/left/image_rect_color","/zed2/right/image_rect_color","/rslidar_points"]

def filter_cloud(cloud):
    min_mask = cloud >= [-3.2,-6.3,0.0]
    max_mask = cloud <= [3.2,0.1,6.4]
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask
    filtered_cloud = cloud[filter_mask]

    # filter by camera view
    c_u = 447.59914779663086
    c_v = 255.83612823486328
    f_u = 365.68
    f_v = 365.68
    baseline = 0.12
    image_shape = (880, 495)


    uv_depth = np.zeros((len(filtered_cloud), 2))
    uv_depth[:,0] = (filtered_cloud[:,0]*f_u)/filtered_cloud[:,2] + c_u
    uv_depth[:,1] = (filtered_cloud[:,1]*f_v)/filtered_cloud[:,2] + c_v

    min_mask = uv_depth >= [0,0]
    max_mask = uv_depth <= [880,495]
    min_mask = min_mask[:, 0] & min_mask[:, 1] 
    max_mask = max_mask[:, 0] & max_mask[:, 1]
    filter_mask = min_mask & max_mask
    filtered_cloud = filtered_cloud[filter_mask]

    return filtered_cloud

def calc_voxel_grid(filtered_cloud, voxel_size):
    xyz_q = np.floor(np.array(filtered_cloud/voxel_size)).astype(int) # quantized point values, here you will loose precision
    vox_grid = np.zeros((int(6.4/voxel_size), int(6.4/voxel_size), int(6.4/voxel_size))) #Empty voxel grid
    offsets = np.array([32, 63, 0])
    xyz_offset_q = xyz_q+offsets
    vox_grid[xyz_offset_q[:,0],xyz_offset_q[:,1],xyz_offset_q[:,2]] = 1 # Setting all voxels containitn a points equal to 1

    xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels
    cloud_np = np.asarray([(pt-offsets)*voxel_size for pt in xyz_v.T])
    return vox_grid, cloud_np

class Bag:
    def listen_image(self, data:Image, side):
        pass
    def listen_pc(self, data:PointCloud2):
        # transform into ZED frame
        if self.mat44 is None:
            self.mat44 = self.listener.asMatrix("zed_left", data.header)
        def xf(p):
            xyz = tuple(np.dot(self.mat44, np.array([p[0], p[1], p[2], 1.0])))[:3]
            return xyz

        point_list = [xf(p) for p in pc2.read_points(data)]
        point_np = np.array(point_list)

        # gt_pcd = o3d.geometry.PointCloud()
        # gt_pcd.points = o3d.utility.Vector3dVector(point_np)

        filtered_cloud = filter_cloud(point_np)
        vox_grid, cloud_np = calc_voxel_grid(filtered_cloud, 0.1)
        
        self.rslidar_points = vox_grid
        self.pub_pc(cloud_np, data.header.stamp)
    
    def pub_pc(self, cloud_np, time):
        points_rgb = np.ones((len(cloud_np), 1))
        color_pl = points_rgb[:, 0] * 65536 * 255
        color_pl = np.expand_dims(color_pl, axis=-1)
        color_pl = color_pl.astype(np.uint32)

        # concat to ROS pointcloud foramt
        concat_pl = np.concatenate((cloud_np, color_pl), axis=1)
        points = concat_pl.tolist()

        # TODO: needs to fix this type conversion
        for i in range(len(points)):
            points[i][3] = int(points[i][3])

        header = Header()
        header.stamp = time
        header.frame_id = "zed_left"

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1),
                ]
        pc = pc2.create_cloud(header, fields, points)

        self.point_cloud_pub.publish(pc)

    def __init__(self) -> None:
        
        self.left_image_raw = None
        self.right_image_raw = None
        self.rslidar_points = None
        self.listener = tf.TransformListener()
        self.listener.waitForTransform("/zed_left", "/robosense", rospy.Time(0),rospy.Duration(1.0))
        self.point_cloud_pub = rospy.Publisher(
            "/voxels", PointCloud2, queue_size=1)
        self.mat44 = None

if __name__ == "__main__":
    rospy.init_node("parse_rosbag", log_level=rospy.INFO)

    bag = Bag()

    rospy.Subscriber("/zed2/left/image_rect_color", Image,
                         bag.listen_image, "left", queue_size=1, buff_size=2**24)
    rospy.Subscriber("/zed2/right/image_rect_color", Image,
                        bag.listen_image, "right", queue_size=1, buff_size=2**24)
    rospy.Subscriber("/rslidar_points", PointCloud2,
                        bag.listen_pc, queue_size=1, buff_size=2**24)

    rospy.spin()