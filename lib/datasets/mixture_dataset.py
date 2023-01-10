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
from datasets.pushing_dataset import PushingDataset
from datasets.tabletop_dataset import TableTopDataset

# in this mixture of datasets, half of each batch is pushing samples
# # the other half is tabletop images
class MixtureDataset(data.Dataset, datasets.imdb):

    def __init__(self, image_set="train", eval=False):

        self._name = 'mixture_object_' + image_set
        self._image_set = image_set
        self._classes_all = ('__background__', 'foreground')
        self._classes = self._classes_all
        self.eval = eval
        self.tabletop = TableTopDataset(image_set=image_set)
        self.pushing = PushingDataset(image_set=image_set)
        self._size = 2 * len(self.pushing)
        np.random.seed(42) 
        # we randomly pick tabletop samples with size of pushing samples
        # self.tabletop_idx = np.random.randint(0, high=len(self.tabletop), size=len(self.pushing))
        # print(self.tabletop_idx[:10])

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        # odd sample is from tabletop
        # even sample is from pushing
        dataset_idx = idx // 2
        if idx % 2 == 0:
            return self.pushing[dataset_idx]
        else:
            return self.tabletop[np.random.randint(0, high=len(self.tabletop), size=1)[0]]



