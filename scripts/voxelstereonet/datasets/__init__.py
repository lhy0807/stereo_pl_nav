from .dataset import VoxelDataset, VoxelKITTIDataset, VoxelDSDataset

__datasets__ = {
    "voxel": VoxelDataset,
    "voxelkitti": VoxelKITTIDataset,
    "voxelds": VoxelDSDataset,
}
