# Copyright (c) 2021. All rights reserved.
from .MSNet2D import MSNet2D
from .MSNet3D import MSNet3D
from .Voxel2D import Voxel2D
from .submodule import model_loss

__models__ = {
    "MSNet2D": MSNet2D,
    "MSNet3D": MSNet3D,
    "Voxel2D": Voxel2D
}
