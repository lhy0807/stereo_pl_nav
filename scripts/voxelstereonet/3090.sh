python train.py --dataset voxelds --datapath /home/chris/pl_ws/src/stereo_pl_nav/datasets/DS \
 --trainlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_train.txt \
 --testlist /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_test.txt \
 --epochs 20 --lrepochs "10,16:2" --logdir /home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/logs \
 --batch_size 4 --test_batch_size 4 --summary_freq 50 --loader_workers 8 \
 --cost_vol_type voxel