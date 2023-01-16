#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


./tools/test_image_with_ms_transformer.py  \
--imgdir data/demo   \
--color *-color.png   \
--depth *-depth.png \
--cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml \
--pretrained data/checkpoints/rgb_pretrain/norm_RGB_pretrained.pth \
--pretrained_crop data/checkpoints/rgb_pretrain/crop_RGB_pretrained.pth \
--network_cfg MSMFormer/configs/mixture_ResNet50.yaml \
--network_crop_cfg MSMFormer/configs/crop_mixture_ResNet50.yaml  \
--input_image COLOR