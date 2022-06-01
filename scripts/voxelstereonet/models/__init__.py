# Copyright (c) 2021. All rights reserved.
from .Voxel2D import Voxel2D
from .Voxel2D_lite import Voxel2D as Voxel2DLite
from .submodule import model_loss, calc_IoU

__models__ = {
    "Voxel2D": Voxel2D,
    "Voxel2D_lite": Voxel2DLite
}
