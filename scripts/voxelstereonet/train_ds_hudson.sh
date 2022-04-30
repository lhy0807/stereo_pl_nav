#!/bin/sh
python train.py --dataset voxelds --datapath /DS \
 --trainlist /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_train.txt \
 --testlist /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_test.txt \
 --epochs 20 --lrepochs "10,16:2" --logdir /scratch/yw3076/lhy/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 36 --test_batch_size 16 --summary_freq 32 --loader_workers 8 --lr 1e-3 --log_folder_suffix odds