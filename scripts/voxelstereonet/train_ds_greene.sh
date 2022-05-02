#!/bin/sh
python train.py --dataset voxelds --datapath /DS \
 --trainlist /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_train.txt \
 --testlist /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_test.txt \
 --epochs 10 --lrepochs "10,16:2" --logdir /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 16 --test_batch_size 16 --summary_freq 50 --loader_workers 8 --lr 1e-3 --cost_vol_type front