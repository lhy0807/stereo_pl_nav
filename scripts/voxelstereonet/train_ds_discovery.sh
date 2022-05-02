#!/bin/sh
python train.py --dataset voxelds --datapath /work/riverlab/hongyu/dataset/DS \
 --trainlist /work/riverlab/hongyu/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_train.txt \
 --testlist /work/riverlab/hongyu/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_test.txt \
 --epochs 10 --lrepochs "10,16:2" --logdir /work/riverlab/hongyu/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 16 --test_batch_size 16 --summary_freq 50 --loader_workers 8