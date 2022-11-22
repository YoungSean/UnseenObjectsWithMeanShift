import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

from meanshiftformer.config import add_meanshiftformer_config
from detectron2.config import get_cfg
from tabletop_config import add_tabletop_config
from detectron2.projects.deeplab import add_deeplab_config
from .topk_test_utils import Predictor_RGBD, get_confident_instances, combine_masks

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import cv2
import scipy.io
import glob
import json

import _init_paths
from fcn.test_dataset import test_sample, filter_labels_depth
from fcn.config import cfg, cfg_from_file, get_output_dir
import networks
from utils.blob import pad_im
from utils import mask as util_

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def get_predictor(input_image="RGBD_ADD"):
# build model
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    #cfg_file = "/home/xy/yxl/UnseenObjectClusteringYXL/MSMFormer/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
    cfg_file = "../../MSMFormer/configs/coco_ms/instance-segmentation/tabeltop_pretrained.yaml"
    cfg.merge_from_file(cfg_file)
    add_tabletop_config(cfg)
    cfg.SOLVER.IMS_PER_BATCH = 1 #
    # cfg.MODEL.WEIGHTS = "/home/xy/yxl/UnseenObjectClusteringYXL/MSMFormer/output_RGB/model_0004999.pth"
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 7
    cfg.INPUT.INPUT_IMAGE = input_image
    # arguments frequently tuned
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    weight_path = "../../MSMFormer/output_0923_kappa30/model_0139999.pth"
    #cfg.device = "cuda:0"
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Predictor_RGBD(cfg)
    return predictor, cfg

def get_backbone():
    num_classes = 2
    pretrained = "/home/xy/yxl/UnseenForMeanShift/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth"
    # pretrained_crop = "/home/xy/yxl/UnseenForMeanShift/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth"
    network_name = "seg_resnet34_8s_embedding"
    if pretrained:
        network_data = torch.load(pretrained)
        # print("=> using pre-trained network '{}'".format(pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda()
    # network = torch.nn.DataParallel(network, device_ids=[0]).cuda()
    # cudnn.benchmark = True
    # network.eval()
    for param in network.parameters():
        param.requires_grad = False

    return network

def get_backbone_crop():
    num_classes = 2
    pretrained_crop = "/home/xy/yxl/UnseenForMeanShift/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth"
    network_name = "seg_resnet34_8s_embedding"

    if pretrained_crop:
        network_data_crop = torch.load(pretrained_crop)
        network_crop = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda()#cuda(device=cfg.device)
        # network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda() #cuda(device=cfg.device)
        # network_crop.eval()
    else:
        network_crop = None

    cudnn.benchmark = True
    for param in network_crop.parameters():
        param.requires_grad = False

    return network_crop

def get_crop_sample_and_features(image, depth, label, topk=True, confident_score=0.9, low_threshold=0.4):
    predictor, _ = get_predictor()
    network_crop = get_backbone_crop()
    sample = {}
    sample["image"] = image
    sample["depth"] = depth
    sample["label"] = label

    outputs = predictor(sample)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score,
                                                  num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=low_threshold)
    binary_mask = combine_masks(confident_instances)


    out_label = torch.as_tensor(binary_mask).unsqueeze(dim=0).cuda()

    if len(depth.shape) == 3:
        depth = torch.unsqueeze(depth, dim=0)
    if len(image.shape) == 3:
        image = torch.unsqueeze(image, dim=0)
    if depth is not None:
        # filter labels on zero depth
        out_label = filter_labels_depth(out_label, depth, 0.8)

    # zoom in refinement
    out_label_refined = None
    if network_crop is not None:
        rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
        if rgb_crop.shape[0] > 0:
            features_crop = network_crop(rgb_crop, out_label_crop, depth_crop)
            labels_crop, selected_pixels_crop = clustering_features(features_crop, num_seeds=num_of_ms_seed)
            # result_crop = cluster_crop(rgb_crop, depth_crop, features_crop)
            # confident_instances_crop = get_confident_instances(result_crop, topk=topk, score=confident_score,
            #                                               num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            #                                               low_threshold=low_threshold)
            # binary_mask_crop = combine_masks(confident_instances_crop)
            # labels_crop = torch.as_tensor(binary_mask_crop).unsqueeze(dim=0).cuda()
            out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)
# save data
def save_data(file_rgb, out_label_refined, roi, features_crop):

    # meta data
    '''
    meta = {'roi': roi, 'features': features_crop.cpu().detach().numpy(), 'labels': out_label_refined.cpu().detach().numpy()}
    filename = file_rgb[:-9] + 'meta.mat'
    scipy.io.savemat(filename, meta, do_compression=True)
    print('save data to {}'.format(filename))
    '''

    # segmentation labels
    label_save = out_label_refined.cpu().detach().numpy()[0]
    label_save = np.clip(label_save, 0, 1) * 255
    label_save = label_save.astype(np.uint8)
    filename = file_rgb[:-4] + '-label.png'
    cv2.imwrite(filename, label_save)
    print('save data to {}'.format(filename))


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = util_.build_matrix_of_indices(height, width)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


def read_sample(filename_color, filename_depth, camera_params):

    # bgr image
    im = cv2.imread(filename_color)

    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        # depth image
        depth_img = cv2.imread(filename_depth, cv2.IMREAD_ANYDEPTH)
        depth = depth_img.astype(np.float32) / 1000.0

        height = depth.shape[0]
        width = depth.shape[1]
        fx = camera_params['fx']
        fy = camera_params['fy']
        px = camera_params['x_offset']
        py = camera_params['y_offset']
        xyz_img = compute_xyz(depth, fx, fy, px, py, height, width)
    else:
        xyz_img = None

    im_tensor = torch.from_numpy(im) / 255.0
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
    im_tensor -= pixel_mean
    image_blob = im_tensor.permute(2, 0, 1)
    sample = {'image_color': image_blob.unsqueeze(0)}

    if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
        sample['depth'] = depth_blob.unsqueeze(0)

    return sample



