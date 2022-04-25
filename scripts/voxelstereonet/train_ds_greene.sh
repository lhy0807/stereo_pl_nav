#!/bin/sh
python train.py --dataset voxelds --datapath /scratch/yw3076/lhy/DS \
 --trainlist /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_train.txt \
 --testlist /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_test.txt \
 --epochs 30 --lrepochs "10,15,20,25:2" --logdir /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 8 --test_batch_size 8 --summary_freq 50 --loader_workers 8