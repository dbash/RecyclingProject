#!/bin/sh

cd ../exper/


#v5_v3

ROOT_DIR=/home/xiaolin/eccv18

CUDA_VISIBLE_DEVICES=0,1 python val_frame.py \
	--arch=inception3_spg \
	--batch_size=1 \
	--img_dir=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/val \
	--num_gpu=1 --dataset=imagenet  \
	--num_classes=2 \
	--snapshot_dir=../snapshots/inception_kitti_full_spg/
