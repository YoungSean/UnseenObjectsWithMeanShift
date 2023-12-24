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

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
def get_backbone():
    num_classes = 2
    # set your filepath for the UCN pretrained checkpoints
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
    cudnn.benchmark = True
    # network.eval()
    for param in network.parameters():
        param.requires_grad = False

    # if pretrained_crop:
    #     network_data_crop = torch.load(pretrained_crop)
    #     network_crop = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda()#cuda(device=cfg.device)
    #     network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda() #cuda(device=cfg.device)
    #     network_crop.eval()
    # else:
    #     network_crop = None

    # if cfg.TEST.VISUALIZE:
    #     index_images = np.random.permutation(len(images_color))
    # else:
    #     index_images = range(len(images_color))

    # for i in index_images:
    #     if os.path.exists(images_color[i]):
    #         print(images_color[i])
    #         # read sample
    #         sample = read_sample(images_color[i], images_depth[i], camera_params)
    #
    #         # run network
    #         out_label, out_label_refined = test_sample(sample, network, network_crop)
    #         # cv2.imshow("image", out_label_refined.squeeze().numpy())
    #         # cv2.waitKey(0)
    #         # # cv2.waitKey(100000)
    #         # cv2.destroyAllWindows()
    #         # print(out_label.shape)
    #     else:
    #         print('files not exist %s' % (images_color[i]))
    return network

def get_backbone_crop():
    num_classes = 2
    # set your filepath for the UCN pretrained checkpoints
    pretrained_crop = "/home/xy/yxl/UnseenForMeanShift/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth"
    network_name = "seg_resnet34_8s_embedding"

    if pretrained_crop:
        network_data_crop = torch.load(pretrained_crop)
        network_crop = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda()#cuda(device=cfg.device)
        network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda() #cuda(device=cfg.device)
        network_crop.eval()
    else:
        network_crop = None

    cudnn.benchmark = True
    for param in network_crop.parameters():
        param.requires_grad = False

    return network_crop
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

#get_backbone()
# if __name__ == "__main__":
#     num_classes = 2
#     pretrained = "/home/xy/yxl/UnseenForMeanShift/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth"
#     pretrained_crop = "/home/xy/yxl/UnseenForMeanShift/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth"
#     network_name = "seg_resnet34_8s_embedding"
#     if pretrained:
#         network_data = torch.load(pretrained)
#         print("=> using pre-trained network '{}'".format(pretrained))
#     else:
#         network_data = None
#         print("no pretrained network specified")
#         sys.exit()
#
#     network = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda()
#     # network = torch.nn.DataParallel(network, device_ids=[0]).cuda()
#     cudnn.benchmark = True
#     network.eval()
#     for param in network.parameters():
#         param.requires_grad = False
#
#     if pretrained_crop:
#         network_data_crop = torch.load(pretrained_crop)
#         network_crop = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda()
#         network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda()
#         network_crop.eval()
#     else:
#         network_crop = None
#     #get images
#     images_color = []
#     filename = os.path.join("/home/xy/yxl/UnseenForMeanShift/data/demo", "*-color.png")
#     files = glob.glob(filename)
#     for i in range(len(files)):
#         filename = files[i]
#         images_color.append(filename)
#     images_color.sort()
#
#     images_depth = []
#     filename = os.path.join("/home/xy/yxl/UnseenForMeanShift/data/demo", "*-depth.png")
#     files = glob.glob(filename)
#     for i in range(len(files)):
#         filename = files[i]
#         images_depth.append(filename)
#     images_depth.sort()
#
#     # check if intrinsics available
#     filename = os.path.join("/home/xy/yxl/UnseenForMeanShift/data/demo", 'camera_params.json')
#     if os.path.exists(filename):
#         with open(filename) as f:
#             camera_params = json.load(f)
#     else:
#         camera_params = None
#
#     cfg.device = "cuda:0"
#     index_images = range(len(images_color))
#     for i in index_images:
#         if os.path.exists(images_color[i]):
#             print(images_color[i])
#             # read sample
#             sample = read_sample(images_color[i], images_depth[i], camera_params)
#             # construct input
#             image = sample['image_color'].cuda()
#             if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
#                 depth = sample['depth'].cuda()
#             else:
#                 depth = None
#
#             if 'label' in sample:
#                 label = sample['label'].cuda()
#             else:
#                 label = None
#
#             # get features
#             #features = network(image, label, depth).detach()
#             out_label, out_label_refined = test_sample(sample, network, network_crop)
#             print(out_label)
# print(images_color)
#features = network(image, label, depth).detach()

