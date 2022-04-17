from .dataset import SceneFlowDataset, KITTIDataset, DrivingStereoDataset, VoxelDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
    "voxel": VoxelDataset,
}
