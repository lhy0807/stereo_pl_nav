#!/bin/sh
python train_amd.py --dataset voxelds --datapath /DS \
 --trainlist /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_train.txt \
 --testlist /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_test.txt \
 --epochs 20 --lrepochs "10,16:2" --logdir /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 8 --test_batch_size 8 --summary_freq 32 --loader_workers 16