#!/bin/sh
python train.py --dataset voxel --datapath /home/chris/pl_ws/src/stereo_pl_nav/datasets/SceneFlow \
 --trainlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/sceneflow_train.txt \
 --testlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/sceneflow_test.txt \
 --epochs 20 --lrepochs "10,12,14,16:2" --logdir /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 3 --test_batch_size 4