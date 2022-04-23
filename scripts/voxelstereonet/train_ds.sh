#!/bin/sh
python train.py --dataset voxelds --datapath /home/chris/pl_ws/src/stereo_pl_nav/datasets/DS \
 --trainlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_train.txt \
 --testlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_test.txt \
 --epochs 30 --lrepochs "10,15,20,25:2" --logdir /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 8 --test_batch_size 8 --summary_freq 50