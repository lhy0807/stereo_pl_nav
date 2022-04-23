#!/bin/sh
python train.py --dataset voxelkitti --datapath /home/chris/pl_ws/src/stereo_pl_nav/datasets/KITTI_2015 \
 --trainlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/kitti15_train.txt \
 --testlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/kitti15_val.txt \
 --epochs 30 --lrepochs "10,15,20,25:2" --logdir /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 2 --test_batch_size 8 --summary_freq 2