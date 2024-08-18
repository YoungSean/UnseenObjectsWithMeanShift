#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

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
from fcn.test_dataset import test_sample
from fcn.config import cfg, cfg_from_file, get_output_dir
import networks
from utils.blob import pad_im
from utils import mask as util_


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--pretrained_crop', dest='pretrained_crop',
                        help='initialize with pretrained checkpoint for crops',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--depth', dest='depth_name',
                        help='depth image pattern',
                        default='*depth.png', type=str)
    parser.add_argument('--color', dest='color_name',
                        help='color image pattern',
                        default='*color.png', type=str)
    parser.add_argument('--imgdir', dest='imgdir',
                        help='path of the directory with the test images',
                        default="data/demo", type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--image_path', dest='image_path',
                        help='path to images', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


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


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = 0
    num_classes = 2
    cfg.MODE = 'TEST'
    print('GPU device {:d}'.format(args.gpu_id))

    # list images
    images_color = []
    filename = os.path.join(args.imgdir, args.color_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_color.append(filename)
    images_color.sort()

    images_depth = []
    filename = os.path.join(args.imgdir, args.depth_name)
    files = glob.glob(filename)
    for i in range(len(files)):
        filename = files[i]
        images_depth.append(filename)
    images_depth.sort()

    # check if intrinsics available
    filename = os.path.join(args.imgdir, 'camera_params.json')
    if os.path.exists(filename):
        with open(filename) as f:
            camera_params = json.load(f)
    else:
        camera_params = None

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    if args.pretrained_crop:
        network_data_crop = torch.load(args.pretrained_crop)
        network_crop = networks.__dict__[args.network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=cfg.device)
        network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        network_crop.eval()
    else:
        network_crop = None

    # cfg.TEST.VISUALIZE = True
    if cfg.TEST.VISUALIZE:
        index_images = np.random.permutation(len(images_color))
    else:
        index_images = range(len(images_color))

    for i in index_images:
        if os.path.exists(images_color[i]):
            print(images_color[i])
            # read sample
            sample = read_sample(images_color[i], images_depth[i], camera_params)

            # run network
            out_label, out_label_refined = test_sample(sample, network, network_crop)
            # cv2.imshow("image", out_label_refined.squeeze().numpy())
            # cv2.waitKey(0)
            # # cv2.waitKey(100000)
            # cv2.destroyAllWindows()
            # print(out_label.shape)
        else:
            print('files not exist %s' % (images_color[i]))
