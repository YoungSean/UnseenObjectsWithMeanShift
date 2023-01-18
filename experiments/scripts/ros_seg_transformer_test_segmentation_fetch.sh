#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

outdir="data/checkpoints"

./ros/test_images_segmentation_transformer.py --gpu $1 \
--cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
--pretrained data/checkpoints/rgbd_pretrain/norm_RGBD_pretrained.pth \
--pretrained_crop data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth \
--network_cfg MSMFormer/configs/mixture_UCN.yaml \
--network_crop_cfg MSMFormer/configs/crop_mixture_UCN.yaml \
--input_image RGBD_ADD \
--camera Fetch \
#--no_refinement     # comment this out if you want to use labe refinement
