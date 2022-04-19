#!/bin/sh
python train.py --dataset voxel --datapath /work/riverlab/hongyu/dataset/SceneFlow \
 --trainlist /work/riverlab/hongyu/stereo_pl_nav/scripts/voxelstereonet/filenames/sceneflow_train.txt \
 --testlist /work/riverlab/hongyu/stereo_pl_nav/scripts/voxelstereonet/filenames/sceneflow_test.txt \
 --epochs 10 --lrepochs "10,12,14,16:2" --logdir /work/riverlab/hongyu/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 8 --test_batch_size 8