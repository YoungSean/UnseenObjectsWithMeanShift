# Author: Yangxiao Lu
# This script is used to add bounding boxes and masks for objects in the dataset.

import torch
import torch.utils.data as data
import os, math
import sys
import time
import random
import numpy as np
import numpy.random as npr
import cv2
import glob
import matplotlib.pyplot as plt
import datasets
import scipy.io
from fcn.config import cfg
from utils.blob import chromatic_transform, add_noise
from utils import augmentation
from utils import mask as util_
from detectron2.structures import BoxMode
import pycocotools
from pathlib import Path
from detectron2.data import detection_utils as utils
from torchvision import transforms

data_loading_params = {

    # Camera/Frustum parameters
    'img_width': 640,
    'img_height': 480,
    'near': 0.01,
    'far': 100,
    'fov': 45,  # vertical field of view in degrees

    'use_data_augmentation': True,

    # Multiplicative noise
    'gamma_shape': 1000.,
    'gamma_scale': 0.001,

    # Additive noise
    'gaussian_scale': 0.005,  # 5mm standard dev
    'gp_rescale_factor': 4,

    # Random ellipse dropout
    'ellipse_dropout_mean': 10,
    'ellipse_gamma_shape': 5.0,
    'ellipse_gamma_scale': 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean': 15,
    'gradient_dropout_alpha': 2.,
    'gradient_dropout_beta': 5.,

    # Random pixel dropout
    'pixel_dropout_alpha': 1.,
    'pixel_dropout_beta': 10.,
}

im_normalization = transforms.Normalize(
    # for RGB
    # mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225]
    # for BGR
    mean=[0.406, 0.456, 0.485],
    std=[0.225, 0.224, 0.229]
)

im_transform = transforms.Compose([
    transforms.ToTensor(),
    im_normalization,
])

def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img

def mask_to_tight_box(mask):
    a = np.transpose(np.nonzero(mask))
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max

# def getPushingDataset(image_set='train'):
#     dataset = TableTopDataset(image_set=image_set)
#     print("The size of the dataset is ", len(dataset))
#     dataset_dicts = []
#     for i in range(len(dataset)):
#         dataset_dicts.append(dataset[i])
#
#     return dataset_dicts

class PushingDataset(data.Dataset, datasets.imdb):

    def __init__(self, image_set="train", pushing_object_path = None, eval=False):

        self._name = 'pushing_object_' + image_set
        self._image_set = image_set
        self._pushing_object_path = self._get_default_path() if pushing_object_path is None \
                            else pushing_object_path
        self._classes_all = ('__background__', 'foreground')
        # self._classes_all = (
        #     "__background__ ",
        #     "table",
        #     "object",
        # )
        self._classes = self._classes_all
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        self.params = data_loading_params
        self.eval = eval

        # get a list of all scenes
        if image_set == 'train':
            data_path = os.path.join(self._pushing_object_path, 'training_set')
            self._pushing_object_path = data_path
        elif image_set == 'test':
            data_path = os.path.join(self._pushing_object_path, 'test_set')
            # self.data_path = sorted(glob.glob(data_path + '/*'))
            self._pushing_object_path = data_path
        elif image_set == 'all':
            data_path = os.path.join(self._pushing_object_path, 'training_set')
            # scene_dirs_train = sorted(glob.glob(data_path + '/*'))
            data_path = os.path.join(self._pushing_object_path, 'test_set')
            # scene_dirs_test = sorted(glob.glob(data_path + '/*'))
            # self.data_path = scene_dirs_train + scene_dirs_test
            self._pushing_object_path = data_path


        self.image_paths = self.list_dataset()

        print('%d images for dataset %s' % (len(self.image_paths), self._name))
        self._size = len(self.image_paths)

        assert os.path.exists(self._pushing_object_path), \
                'pushing_object path does not exist: {}'.format(self._pushing_object_path)

    def process_depth(self, depth_img):
        """ Process depth channel
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # add random noise to depth
        if self.params['use_data_augmentation']:
            depth_img = augmentation.add_noise_to_depth(depth_img, self.params)
            depth_img = augmentation.dropout_random_ellipses(depth_img, self.params)

        # Compute xyz ordered point cloud and add noise
        xyz_img = compute_xyz(depth_img, self.params)
        if self.params['use_data_augmentation']:
            xyz_img = augmentation.add_noise_to_xyz(xyz_img, depth_img, self.params)
        return xyz_img

    def process_label_to_annos(self, labels):
        """ Process labels
                - Map the labels to [H x W x num_instances] numpy array
        """
        H, W = labels.shape

        # Find the unique (nonnegative) labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(labels)

        # Drop 0 if it's in there
        if unique_nonnegative_indices[0] == 0:
            unique_nonnegative_indices = unique_nonnegative_indices[1:]
        num_instances = unique_nonnegative_indices.shape[0]
        # NOTE: IMAGES WITH BACKGROUND ONLY HAVE NO INSTANCES

        # Get binary masks
        binary_masks = np.zeros((H, W, num_instances), dtype=np.float32)
        for i, label in enumerate(unique_nonnegative_indices):
            binary_masks[..., i] = (labels == label).astype(np.float32)

        # Get bounding boxes
        boxes = np.zeros((num_instances, 4))
        for i in range(num_instances):
            boxes[i, :] = np.array(mask_to_tight_box(binary_masks[..., i]))

        # Get labels for each mask
        labels = unique_nonnegative_indices.clip(1, 2)

        # Turn them into torch tensors
        boxes = augmentation.array_to_tensor(boxes)
        binary_masks = augmentation.array_to_tensor(binary_masks)
        labels = augmentation.array_to_tensor(labels).long()

        return boxes, binary_masks, labels

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
        # print("mapped labels", np.unique(mapped_labels))
        return foreground_labels

    def pad_crop_resize(self, img, label, depth):
        """ Crop the image around the label mask, then resize to 224x224
        """

        H, W, _ = img.shape

        # sample an object to crop
        K = np.max(label)
        while True:
            if K > 0:
                idx = np.random.randint(1, K+1)
            else:
                idx = 0
            foreground = (label == idx).astype(np.float32)

            # get tight box around label/morphed label
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
            # cv2.imshow("foreground", foreground*100)
            # cv2.waitKey(0)
            x_min, y_min, x_max, y_max = util_.mask_to_tight_box(foreground)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            # make bbox square
            x_delta = x_max - x_min
            y_delta = y_max - y_min
            if x_delta > y_delta:
                y_min = cy - x_delta / 2
                y_max = cy + x_delta / 2
            else:
                x_min = cx - y_delta / 2
                x_max = cx + y_delta / 2

            sidelength = x_max - x_min
            padding_percentage = np.random.uniform(cfg.TRAIN.min_padding_percentage, cfg.TRAIN.max_padding_percentage)
            padding = int(round(sidelength * padding_percentage))
            if padding == 0:
                padding = 25

            # Pad and be careful of boundaries
            x_min = max(int(x_min - padding), 0)
            x_max = min(int(x_max + padding), W-1)
            y_min = max(int(y_min - padding), 0)
            y_max = min(int(y_max + padding), H-1)

            # crop
            if (y_min == y_max) or (x_min == x_max):
                continue

            img_crop = img[y_min:y_max+1, x_min:x_max+1]
            label_crop = label[y_min:y_max+1, x_min:x_max+1]
            roi = [x_min, y_min, x_max, y_max]
            if depth is not None:
                depth_crop = depth[y_min:y_max+1, x_min:x_max+1]
            break

        # resize
        s = cfg.TRAIN.SYN_CROP_SIZE
        img_crop = cv2.resize(img_crop, (s, s))
        label_crop = cv2.resize(label_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            depth_crop = cv2.resize(depth_crop, (s, s), interpolation=cv2.INTER_NEAREST)
        else:
            depth_crop = None

        return img_crop, label_crop, depth_crop

    # sample num of pixel for clustering instead of using all
    def sample_pixels(self, labels, num=1000):
        # -1 ignore
        labels_new = -1 * np.ones_like(labels)
        K = np.max(labels)
        for i in range(K+1):
            index = np.where(labels == i)
            n = len(index[0])
            if n <= num:
                labels_new[index[0], index[1]] = i
            else:
                perm = np.random.permutation(n)
                selected = perm[:num]
                labels_new[index[0][selected], index[1][selected]] = i
        return labels_new

    def list_dataset(self):
        data_path = Path(self._pushing_object_path)
        # print('data path', data_path)
        seqs = sorted(list(Path(data_path).glob('*/*T*')))
        # print(seqs)

        image_paths = []
        for seq in seqs:
            paths = sorted(list((seq).glob('color*.jpg')))
            image_paths += paths
        return image_paths

    def __getitem__(self, idx):
        # BGR image
        filename = str(self.image_paths[idx])
        # print("file name", filename)
        im = cv2.imread(filename)

        # meta data
        meta_filename = filename.replace('color', 'meta').replace('jpg', 'mat')
        data = scipy.io.loadmat(meta_filename)

        # Label
        labels_filename = filename.replace('color', 'label-final').replace('jpg', 'png')
        foreground_labels = cv2.imread(labels_filename, cv2.IMREAD_GRAYSCALE)
        # unique_indices = np.unique(foreground_labels)
        # print('unique masks', unique_indices)
        boxes, binary_masks, labels = self.process_label_to_annos(foreground_labels)
        foreground_labels = self.process_label(foreground_labels)
        # plt.imshow(foreground_labels * 50)
        # plt.show()


        # boxes.shape: [num_instances x 4], binary_masks.shape: [num_instances x H x W], labels.shape: [num_instances]

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            # Depth image
            depth_img_filename = filename.replace('color', 'depth').replace('jpg', 'png')
            depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH).astype(np.float32) # This reads a 16-bit single-channel image. Shape: [H x W]
            # get xyz_img
            height = depth_img.shape[0]
            width = depth_img.shape[1]
            factor_depth = data['factor_depth']
            intrinsics = data['intrinsic_matrix']
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            px = intrinsics[0, 2]
            py = intrinsics[1, 2]

            depth_img /= factor_depth
            xyz_img = compute_xyz(depth_img, fx, fy, px, py, height, width)
            # plt.imshow(xyz_img)
            # plt.show()
        else:
            xyz_img = None

        # crop
        if cfg.TRAIN.SYN_CROP and cfg.MODE == 'TRAIN':
            #print(boxes)
            im, foreground_labels, xyz_img = self.pad_crop_resize(im, foreground_labels, xyz_img)
            foreground_labels = self.process_label(foreground_labels)
            boxes, binary_masks, labels = self.process_label_to_annos(foreground_labels)

        # sample labels
        if cfg.TRAIN.EMBEDDING_SAMPLING:
            foreground_labels = self.sample_pixels(foreground_labels, cfg.TRAIN.EMBEDDING_SAMPLING_NUM)
        if cfg.TRAIN.CHROMATIC and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = chromatic_transform(im)
        if cfg.TRAIN.ADD_NOISE and cfg.MODE == 'TRAIN' and np.random.rand(1) > 0.1:
            im = add_noise(im)

        record = {}
        record["raw_image"] = torch.from_numpy(im).permute(2, 0, 1)
        record["raw_depth"] = xyz_img
        record["file_name"] = filename
        record["image_id"] = idx

        #torch.permute(torch.from_numpy(xyz_img), (2,0,1))
        objs = []
        # get annotations
        for box, mask, label in zip(boxes, binary_masks, labels):
            obj = {
                "bbox": box.numpy(),
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": pycocotools.mask.encode(np.asarray(mask.to(torch.uint8), order="F")),
                "category_id": 1,
            }
            objs.append(obj)
        record["annotations"] = objs

        # obtain label tensor
        label_blob = torch.from_numpy(foreground_labels).unsqueeze(0)
        record["label"] = label_blob
        # get RGB tensor
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor -= self._pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)

        # plt.imshow(im_rgb)
        # plt.show()
        # get RGB tensor
        # im_rgb = im #cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im_tensor = im_transform(im_rgb)
        # image_blob = im_tensor
        record['image_color'] = image_blob
        record["height"] = image_blob.shape[-2]
        record["width"] = image_blob.shape[-1]

        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
            depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
            record['depth'] = depth_blob
        # record["image"] = torch.permute(torch.from_numpy(im), (2, 0, 1))
        # print("label shape",label_blob.shape)
        return record

    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where tabletop_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'pushing_data')
        # return os.path.join(datasets.ROOT_DIR, 'data', 'tabletop_demo')

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.params['img_height'],
                "width": self.params['img_width'],
                "idx": idx
               }
