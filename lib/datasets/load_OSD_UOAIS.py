import cv2
import numpy as np
import torch


def normalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ normalize the input depth (mm) and return depth image (0 ~ 255)
    Args:
        depth ([np.float]): depth array [H, W] (mm)
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.

    Returns:
        [np.uint8]: normalized depth array [H, W, 3] (0 ~ 255)
    """
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))
    return depth


def unnormalize_depth(depth, min_val=250.0, max_val=1500.0):
    """ unnormalize the input depth (0 ~ 255) and return depth image (mm)
    Args:
        depth([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        min_val (float, optional): [min depth]. Defaults to 250 mm
        max_val (float, optional): [max depth]. Defaults to 1500 mm.
    Returns:
        [np.float]: depth array [H, W] (mm)
    """
    depth = np.float32(depth) / 255
    depth = depth * (max_val - min_val) + min_val
    return depth


def inpaint_depth(depth, factor=1, kernel_size=3, dilate=False):
    """ inpaint the input depth where the value is equal to zero

    Args:
        depth ([np.uint8]): normalized depth array [H, W, 3] (0 ~ 255)
        factor (int, optional): resize factor in depth inpainting. Defaults to 4.
        kernel_size (int, optional): kernel size in depth inpainting. Defaults to 5.

    Returns:
        [np.uint8]: inpainted depth array [H, W, 3] (0 ~ 255)
    """

    H, W, _ = depth.shape
    resized_depth = cv2.resize(depth, (W // factor, H // factor))
    mask = np.all(resized_depth == 0, axis=2).astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    inpainted_data = cv2.inpaint(resized_depth, mask, kernel_size, cv2.INPAINT_TELEA)
    inpainted_data = cv2.resize(inpainted_data, (W, H))
    depth = np.where(depth == 0, inpainted_data, depth)
    return depth


def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4:  # NHWC
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    elif array.ndim == 3:  # HWC
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    else:  # everything else
        tensor = torch.from_numpy(array).float()

    return tensor


def standardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes
        @return: a [H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[..., i] = (image[..., i] / 255. - mean[i]) / std[i]

    return image_standardized

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.utils.data as data
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import datasets
import open3d

from fcn.config import cfg
from utils.blob import chromatic_transform, add_noise
from utils import mask as util_
import imageio


class OSDObject_UOAIS(data.Dataset, datasets.imdb):
    def __init__(self, image_set, osd_object_path = None):

        self._name = 'osd_object_' + image_set
        self._image_set = image_set
        self._osd_object_path = self._get_default_path() if osd_object_path is None \
                            else osd_object_path
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        self._width = 640
        self._height = 480

        # get all images
        data_path = os.path.join(self._osd_object_path, 'image_color')
        self.image_files = sorted(glob.glob(data_path + '/*.png'))

        print('%d images for dataset %s' % (len(self.image_files), self._name))
        self._size = len(self.image_files)
        assert os.path.exists(self._osd_object_path), \
                'osd_object path does not exist: {}'.format(self._osd_object_path)


    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels


    def __getitem__(self, idx):

        # BGR image
        filename = self.image_files[idx]
        im = cv2.imread(filename)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
        #     im = chromatic_transform(im)
        # if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
        #     im = add_noise(im)
        # im_tensor = torch.from_numpy(im) / 255.0

        # im_tensor_bgr = im_tensor.clone()
        # im_tensor_bgr = im_tensor_bgr.permute(2, 0, 1)

        # im_tensor -= self._pixel_mean
        # image_blob = im_tensor.permute(2, 0, 1)

        # Label
        labels_filename = filename.replace('image_color', 'annotation')
        foreground_labels = util_.imread_indexed(labels_filename)
        foreground_labels = self.process_label(foreground_labels)
        label_blob = torch.from_numpy(foreground_labels).unsqueeze(0)

        index = filename.find('OSD')
        # use coco mean and std
        # if cfg.INPUT == 'COLOR':
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image_blob = (torch.from_numpy(im).permute(2, 0, 1) - torch.Tensor([123.675, 116.280, 103.530]).view(-1, 1, 1).float()) / torch.Tensor([58.395, 57.120, 57.375]).view(-1, 1, 1).float()
        sample = {'image_color': image_blob,
                #   'image_color_bgr': im_tensor_bgr,
                  'label': label_blob,
                  'filename': filename[index+4:],
                  'file_name': filename,
                  # 'raw_image': torch.from_numpy(im).permute(2, 0, 1)
                  }

        # Depth image
        # different way to deal with depth image
        # https://github.com/gist-ailab/uoais/blob/main/eval/eval_utils.py#LL47C84-L47C84
        depth_path= filename.replace('image_color', 'disparity')
        depth_img = imageio.imread(depth_path)
        depth_img = normalize_depth(depth_img)
        depth_img = inpaint_depth(depth_img) / 255.0 # range [-1, 1]
        # print(depth_img.shape)
        depth_img = torch.from_numpy(depth_img).permute(2, 0, 1).float()
        sample['depth'] = depth_img

        # if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        #     pcd_filename = filename.replace('image_color', 'pcd')
        #     pcd_filename = pcd_filename.replace('png', 'pcd')
        #     pcd = open3d.io.read_point_cloud(pcd_filename)
        #     pcloud = np.asarray(pcd.points).astype(np.float32)
        #     pcloud[np.isnan(pcloud)] = 0
        #     xyz_img = pcloud.reshape((self._height, self._width, 3))
        #     depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
        #     sample['depth'] = depth_blob

        return sample


    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where osd_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'OSD')
