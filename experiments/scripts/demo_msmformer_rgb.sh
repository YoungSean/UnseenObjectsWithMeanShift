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
--pretrained data/checkpoints/RGB_norm_model_0069999.pth \
--pretrained_crop data/checkpoints/RGB_crop_model_final.pth \
--network_cfg MSMFormer/configs/tabletop_pretrained.yaml \
--network_crop_cfg MSMFormer/configs/crop_tabletop_pretrained.yaml
--input_image COLOR
