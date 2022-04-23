from .dataset import SceneFlowDataset, KITTIDataset, DrivingStereoDataset, VoxelDataset, VoxelKITTIDataset, VoxelDSDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
    "voxel": VoxelDataset,
    "voxelkitti": VoxelKITTIDataset,
    "voxelds": VoxelDSDataset,
}
