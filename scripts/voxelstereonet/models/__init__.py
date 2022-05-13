# Copyright (c) 2021. All rights reserved.
from .Voxel2D import Voxel2D
from .submodule import model_loss, calc_IoU

__models__ = {
    "Voxel2D": Voxel2D
}
