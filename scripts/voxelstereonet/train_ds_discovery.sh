#!/bin/sh
python train.py --dataset voxelds --datapath /work/riverlab/hongyu/dataset/DS \
 --trainlist /work/riverlab/hongyu/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_train.txt \
 --testlist /work/riverlab/hongyu/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_test.txt \
 --epochs 30 --lrepochs "10,15,20,25:2" --logdir /work/riverlab/hongyu/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 8 --test_batch_size 8 --summary_freq 32 --loader_workers 8