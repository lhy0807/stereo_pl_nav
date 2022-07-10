# Copyright (c) 2021. All rights reserved.
from .Voxel2D import Voxel2D
from .Voxel2D_lite import Voxel2D as Voxel2DLite
from .Voxel2D_sparse import Voxel2D as Voxel2DSparse
from .submodule import model_loss, calc_IoU

__models__ = {
    "Voxel2D": Voxel2D,
    "Voxel2D_lite": Voxel2DLite,
    "Voxel2D_sparse": Voxel2DSparse,
}
