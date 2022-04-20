#!/bin/sh
python train.py --dataset voxel --datapath /home/chris/pl_ws/src/stereo_pl_nav/datasets/SceneFlow \
 --trainlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/sceneflow_train.txt \
 --testlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/sceneflow_test.txt \
 --epochs 30 --lrepochs "5,10,15,20,25:2" --logdir /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 2 --test_batch_size 2