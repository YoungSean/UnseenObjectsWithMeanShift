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
--pretrained data/checkpoints/rgb_finetuned/norm_RGB_finetuned_data04_6epoch.pth \
--pretrained_crop data/checkpoints/rgb_finetuned/crop_RGB_finetuned_all_5epoch.pth \
--network_cfg MSMFormer/configs/mixture_ResNet50.yaml \
--network_crop_cfg MSMFormer/configs/crop_mixture_ResNet50.yaml  \
--input_image COLOR
