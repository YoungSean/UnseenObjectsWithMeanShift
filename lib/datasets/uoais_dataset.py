# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np

from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager


from detectron2.data import MetadataCatalog, DatasetCatalog
from tqdm import tqdm
import pycocotools.mask as mask_utils

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
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
import detectron2.data.datasets  # noqa # add pre-defined metadata
import sys
import imageio

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_uoais_json"]

MetadataCatalog.get("tabletop_object_train").set(thing_classes=['__background__', 'object'])

def load_segm(anno, type):
    segm = anno.get(type, None)
    if isinstance(segm, dict):
        if isinstance(segm["counts"], list):
            # convert to compressed RLE
            segm = mask_util.frPyObjects(segm, *segm["size"])
    else:
        # filter out invalid polygons (< 3 points)
        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
        if len(segm) == 0:
            num_instances_without_valid_segmentation += 1
            segm = None
    return segm

class  UOAIS_Dataset(data.Dataset, datasets.imdb):
    def __init__(self, image_set="train", uoais_object_path = None, eval=False):
        from pycocotools.coco import COCO
        timer = Timer()
        self._name = 'uoais_object_' + image_set
        self._image_set = image_set
        self._uoais_object_path = self._get_default_path() if uoais_object_path is None \
                            else uoais_object_path
        self._classes_all = ('__background__', 'foreground')
        self._json_file = os.path.join(self._uoais_object_path, "annotations", "coco_anns_uoais_sim_{}.json".format(image_set))

        # get a list of all scenes
        if image_set == 'train':
            self._image_root = os.path.join(self._uoais_object_path, 'train')
        elif image_set == 'val':
            self._image_root = os.path.join(self._uoais_object_path, 'val')

        elif image_set == 'all':
            pass

        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(self._json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(self._json_file, timer.seconds()))

        self._coco_api = coco_api
        json_file = self._json_file
        # sort indices for reproducible results
        img_ids = sorted(coco_api.imgs.keys())
        # imgs is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}
        imgs = coco_api.loadImgs(img_ids)
        # anns is a list[list[dict]], where each dict is an annotation
        # record for an object. The inner list enumerates the objects in an image
        # and the outer list enumerates over images. Example of anns[0]:
        # [{'segmentation': [[192.81,
        #     247.09,
        #     ...
        #     219.03,
        #     249.06]],
        #   'area': 1035.749,
        #   'iscrowd': 0,
        #   'image_id': 1268,
        #   'bbox': [192.81, 224.8, 74.73, 33.43],
        #   'category_id': 16,
        #   'id': 42986},
        #  ...]
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        total_num_valid_anns = sum([len(x) for x in anns])
        total_num_anns = len(coco_api.anns)
        if total_num_valid_anns < total_num_anns:
            logger.warning(
                f"{json_file} contains {total_num_anns} annotations, but only "
                f"{total_num_valid_anns} of them match to images in the file."
            )

        if "minival" not in json_file:
            # The popular valminusminival & minival annotations for COCO2014 contain this bug.
            # However the ratio of buggy annotations there is tiny and does not affect accuracy.
            # Therefore we explicitly white-list them.
            ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
            assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
                json_file
            )

        imgs_anns = list(zip(imgs, anns))
        # only use the TableTop part
        imgs_anns = imgs_anns[22500:]
        self.imgs_anns = imgs_anns
        logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

        dataset_dicts = []



        self._classes = self._classes_all
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        # self._width = 640
        # self._height = 480
        # self.image_paths = self.list_dataset()
        # self.eval = eval

        # print('%d images for dataset %s' % (len(self.image_paths), self._name))
        # self._size = len(self.image_paths)
        # self.max_num_object = 0
        assert os.path.exists(self._uoais_object_path), \
                'uoais_object path does not exist: {}'.format(self._uoais_object_path)
        assert os.path.exists(self._json_file), \
                'Path does not exist: {}'.format(self._json_file)
        # assert os.path.exists(self._image_root), \
        #         'Path does not exist: {}'.format(self._image_root)
        # self._anno = self._load_uoais_object_annotation()
        # self._image_index = self._load_image_set_index()

    def __getitem__(self, idx):
        record = {}
        ann_keys = ["iscrowd", "bbox", "keypoints", "category_id", ]  # + (extra_annotation_keys or [])
        num_instances_without_valid_segmentation = 0

        img_dict = self.imgs_anns[idx][0]
        anno_dict_list = self.imgs_anns[idx][1]
        file_name = os.path.join(self._image_root, img_dict["file_name"])
        im = cv2.imread(file_name)
        # visualize the color image
        # color = im.copy()
        # color = np.ascontiguousarray(color[:, :, ::-1])
        # plt.imshow(color)
        # plt.show()

        # deal with the color image
        im_tensor = torch.from_numpy(im) / 255.0
        im_tensor -= self._pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        # use coco mean and std
        if cfg.INPUT == 'COLOR':
            image_blob = (torch.from_numpy(im).permute(2, 0, 1) - torch.Tensor([123.675, 116.280, 103.530]).view(-1, 1,
                                                                                                                 1).float()) / torch.Tensor(
                [58.395, 57.120, 57.375]).view(-1, 1, 1).float()
        record['image_color'] = image_blob
        record["file_name"] = file_name
        record["depth_file_name"] = os.path.join(self._image_root, img_dict["depth_file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        # deal with the depth image
        depth = imageio.imread(record["depth_file_name"]).astype(np.float32)
        # DEPTH_RANGE = [2500, 15000] # mm
        self.depth_max = 15000
        self.depth_min = 2500
        depth[depth > self.depth_max] = self.depth_max
        depth[depth < self.depth_min] = self.depth_min
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min) * 255
        # plt.imshow(depth)
        # plt.show()
        depth = np.expand_dims(depth, -1)
        depth = np.repeat(depth, 3, -1)
        depth = torch.from_numpy(depth).permute(2, 0, 1) / 255.0  # how to deal with the depth image? to do

        record['depth'] = depth
        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            # if anno.get("segmentation", None):  # either list[list[float]] or dict(RLE)
            #     obj["segmentation"] = load_segm(anno, "segmentation")
            # if anno.get("visible_mask", None):
            #     obj["visible_mask"] = load_segm(anno, "visible_mask")
            if anno.get("visible_mask", None):
                obj["segmentation"] = load_segm(anno, "visible_mask")
            if anno.get("occluded_mask", None):
                obj["occluded_mask"] = load_segm(anno, "occluded_mask")
            obj["occluded_rate"] = anno.get("occluded_rate", None)

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = 1  # orginal is 0
            objs.append(obj)
        record["annotations"] = objs
        # dataset_dicts.append(record)
        dirname = "uoais-data-vis"
        os.makedirs(dirname, exist_ok=True)
        meta=MetadataCatalog.get("tabletop_object_train")
        img = np.array(Image.open(record["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        masks = [x["segmentation"] for x in record["annotations"]]
        vis = visualizer.draw_dataset_dict(record)
        masks = visualizer._convert_masks(masks)
        areas = np.asarray([x.area() for x in masks])
        sorted_idxs = np.argsort(-areas).tolist()
        self.masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
        label = self.from_masks_to_label()

        # obtain label tensor
        label_blob = torch.from_numpy(label).unsqueeze(0)
        record["label"] = label_blob
        # plt.imshow(label)
        # plt.show()
        # print("label shape", label.shape)
        # print("label unique", np.unique(label))
        # print('label:', label)

        # plt.imshow(labels[0]['mask'])
        # plt.show()
        # fpath = os.path.join(dirname, os.path.basename(record["file_name"]))
        # vis.save(fpath)
        return record

    def from_masks_to_label(self):
        inital_mask = self.masks[0].mask
        for i in range(len(self.masks)):
            mask = self.masks[i].mask
            inital_mask[mask!=0] = i+1
        return inital_mask

    def _get_default_path(self):
        """
        Return the default path where uoais_object is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'UOAIS_Sim')

    def __len__(self):
        return 100
        #return len(self.imgs_anns)


def load_uoais_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with uoais's instances annotation format.
    For amodal instance segmentation, dataset_name should include the keword "amodal"
    Args:
        json_file (str): full path to the json file in UOA instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:
            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.
            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        # meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        # id_map = {v: i for i, v in enumerate(cat_ids)}
        if id_map is not None:
            meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id", ] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["depth_file_name"] = os.path.join(image_root, img_dict["depth_file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if anno.get("segmentation", None):  # either list[list[float]] or dict(RLE)
                obj["segmentation"] = load_segm(anno, "segmentation")
            if anno.get("visible_mask", None):
                obj["visible_mask"] = load_segm(anno, "visible_mask")
            if anno.get("occluded_mask", None):
                obj["occluded_mask"] = load_segm(anno, "occluded_mask")
            obj["occluded_rate"] = anno.get("occluded_rate", None)

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = 0
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


if __name__ == "__main__":
    """
    Test the uoais json dataset loader.
    Usage:
        python -m detectron2.data.datasets.uoais \
            path/to/json path/to/image_root dataset_name
        "dataset_name" can be "uoais_val", or other
        pre-registered ones
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    dataset = UOAIS_Dataset()
    print(dataset[0]["file_name"])


    # assert sys.argv[3] in DatasetCatalog.list()
    # meta = MetadataCatalog.get(sys.argv[3])
    #
    # dicts = load_uoais_json(sys.argv[1], sys.argv[2], sys.argv[3])
    # logger.info("Done loading {} samples.".format(len(dicts)))
    #
    # dirname = "uoais-data-vis"
    # os.makedirs(dirname, exist_ok=True)
    # for d in tqdm(dicts):
    #     img = np.array(Image.open(d["file_name"]))
    #     visualizer = Visualizer(img, metadata=meta)
    #     vis = visualizer.draw_dataset_dict(d)
    #     fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
    #     vis.save(fpath)