import copy
import logging

import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask
from detectron2.data.transforms.augmentation import Augmentation
from PIL import Image
import cv2
from fvcore.transforms import transform, Transform
import random
from detectron2.data.transforms import RandomCrop, StandardAugInput
from detectron2.structures import BoxMode

__all__ = ["UnseenInstanceDatasetMapper"]

def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)

    # if some instance is cropped extend the box
    if not crop_box:
        num_modifications = 0
        modified = True

        # convert crop_size to float
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                raise ValueError(
                    "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                        len(instances)
                    )
                )
                return Transform.CropTransform(0, 0, image_size[1], image_size[0])

    return Transform.CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))


def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False

    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]

    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    """ Instance-aware cropping.
    """

    def __init__(self, crop_type, crop_size, crop_instance=True):
        """
        Args:
            crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance
        self.input_args = ("image", "boxes")

    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    Modified to support RGB-D image (W, H, 6)
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) == 3 and img.shape[2] == 6:
                depth_image = img[:, :, 3:6]
                pil_image = Image.fromarray(img[:, :, :3])
            elif len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
            elif len(img.shape) == 3 and img.shape[2] == 6:
                depth_image = cv2.resize(depth_image, (self.new_w, self.new_h), cv2.INTER_NEAREST)
                ret = np.concatenate([pil_image, depth_image], -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(
                img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
            )
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)

class Resize(Augmentation):
    """Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        return ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )



class UnseenInstanceDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for instance segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """


    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        self.img_size = [640, 480]
        cr = 0.5 # crop ratio 
        self.augmentation_lists = [
                T.RandomApply(T.RandomCrop("relative_range", (cr, cr))),
                T.RandomFlip(0.5),
                Resize((self.img_size[1], self.img_size[0]))
            ]
        self.augmentation_lists = T.AugmentationList(self.augmentation_lists)
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {self.augmentation_lists}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        ), T.RandomFlip()]

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret


    """
        General dataset mapper that applies pre-processing to datasets before feeding to the model.
        
        For UOAIS dataset, need to use the other call method after line 423.
    """

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "UnseenDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = dataset_dict["image_color"]
        # raw_image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # raw_image = torch.from_numpy(raw_image)
        if "depth" in dataset_dict.keys():
            depth = dataset_dict["depth"]
        else:
            depth = None

        label = dataset_dict["label"]
        # orig_label = orig_label.squeeze().numpy()

        # HxW -> 1xHxW
        # label = np.expand_dims(label, axis=2)
        # label = torch.from_numpy(label)
        # dataset_dict["label"] = label
        # print(orig_label == label)
        # transform instnace masks
        assert "annotations" in dataset_dict
        for anno in dataset_dict["annotations"]:
            anno.pop("keypoints", None)

        annos = [
            # utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            obj
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if len(annos):
            assert "segmentation" in annos[0]
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                    segm.ndim
                )
                # mask array
                masks.append(segm)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a binary segmentation mask "
                    " in a 2D numpy array of shape HxW.".format(type(segm))
                )

        # Pad image and segmentation label here!
        # image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # depth = torch.as_tensor(np.ascontiguousarray(depth.transpose(2, 0, 1)))
        masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

        # label = torch.as_tensor(np.ascontiguousarray(label.transpose(2, 0, 1)))

        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            # pad image
            image = F.pad(image, padding_size, value=128).contiguous()
            if depth:
                depth = F.pad(depth, padding_size, value=128).contiguous()
            # pad label
            label = F.pad(label, padding_size, value=128).contiguous()
            # pad mask
            masks = [F.pad(x, padding_size, value=0).contiguous() for x in masks]

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        dataset_dict["label"] = label.to(torch.float)
        dataset_dict["depth"] = depth
        # dataset_dict["raw_image"] = raw_image

        # Prepare per-category binary masks
        instances = Instances(image_shape)
        instances.gt_classes = classes
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
        else:
            masks = BitMasks(torch.stack(masks))
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances

        return dataset_dict


"""
    call method for UOAIS dataset setting.
    Comment out the original call method for UOAIS format and Use this one instead.
"""
    # def __call__(self, dataset_dict):
    #
    #     """
    #     Args:
    #         dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
    #
    #     Returns:
    #         dict: a format that builtin models in detectron2 accept
    #     """
    #     assert self.is_train, "UnseenDatasetMapper should only be used for training!"
    #
    #     dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    #     image = utils.read_image(
    #         dataset_dict["file_name"], format='RGB'
    #     )
    #
    #     depth = dataset_dict["depth"]
    #     image = np.concatenate([image, depth], -1)
    #
    #     boxes = np.asarray([BoxMode.convert(
    #                 instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
    #                 )
    #                     for instance in dataset_dict["annotations"]])
    #
    #     # apply the color augmentation
    #     aug_input = T.AugInput(image, boxes=boxes)
    #     transforms = self.augmentation_lists(aug_input)
    #     image = aug_input.image
    #     image_shape = image.shape[:2]  # h, w
    #     color = image[:,:,:3]
    #     depth = image[:,:,3:]
    #     # import sys
    #     # sys.exit()
    #
    #
    #     # if "depth" in dataset_dict.keys():
    #     #     depth = dataset_dict["depth"]
    #     # else:
    #     #     depth = None
    #
    #     # label = dataset_dict["label"]
    #
    #     # transform instnace masks
    #     assert "annotations" in dataset_dict
    #     for anno in dataset_dict["annotations"]:
    #         anno.pop("keypoints", None)
    #
    #     annos = [
    #         # transform_instance_annotations(
    #         #         obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
    #         #     )
    #         utils.transform_instance_annotations(obj, transforms, image.shape[:2])
    #         for obj in dataset_dict.pop("annotations")
    #         if obj.get("iscrowd", 0) == 0
    #     ]
    #
    #     if len(annos):
    #         assert "segmentation" in annos[0]
    #     segms = [obj["segmentation"] for obj in annos]
    #     masks = []
    #     for segm in segms:
    #         if isinstance(segm, list):
    #             # polygon
    #             masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
    #         elif isinstance(segm, dict):
    #             # COCO RLE
    #             masks.append(mask_util.decode(segm))
    #         elif isinstance(segm, np.ndarray):
    #             assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
    #                 segm.ndim
    #             )
    #             # mask array
    #             masks.append(segm)
    #         else:
    #             raise ValueError(
    #                 "Cannot convert segmentation of type '{}' to BitMasks!"
    #                 "Supported types are: polygons as list[list[float] or ndarray],"
    #                 " COCO-style RLE as a dict, or a binary segmentation mask "
    #                 " in a 2D numpy array of shape HxW.".format(type(segm))
    #             )
    #
    #     # Pad image and segmentation label here!
    #     # image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    #     # depth = torch.as_tensor(np.ascontiguousarray(depth.transpose(2, 0, 1)))
    #     depth = torch.from_numpy(depth).permute(2, 0, 1) / 255.0
    #     # For RGB and RGBD
    #     dataset_dict["image"] = (torch.from_numpy(color).permute(2, 0, 1) - torch.Tensor([123.675, 116.280, 103.530]).view(-1, 1,
    #                                                                                                              1).float()) / torch.Tensor(
    #             [58.395, 57.120, 57.375]).view(-1, 1, 1).float()
    #     # # for RGBD
    #     # image = torch.from_numpy(image) / 255.0 - torch.tensor(np.array([[[102.9801, 115.9465, 122.7717]]]) / 255.0).float()
    #     # dataset_dict["image"] = image.permute(2, 0, 1)
    #
    #     masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]
    #
    #     # label = torch.as_tensor(np.ascontiguousarray(label.transpose(2, 0, 1)))
    #
    #     classes = [int(obj["category_id"]) for obj in annos]
    #     classes = torch.tensor(classes, dtype=torch.int64)
    #
    #     if self.size_divisibility > 0:
    #         image_size = (image.shape[-2], image.shape[-1])
    #         padding_size = [
    #             0,
    #             self.size_divisibility - image_size[1],
    #             0,
    #             self.size_divisibility - image_size[0],
    #         ]
    #         # pad image
    #         image = F.pad(image, padding_size, value=128).contiguous()
    #         if depth:
    #             depth = F.pad(depth, padding_size, value=128).contiguous()
    #         # pad label
    #         label = F.pad(label, padding_size, value=128).contiguous()
    #         # pad mask
    #         masks = [F.pad(x, padding_size, value=0).contiguous() for x in masks]
    #
    #     image_shape = (image.shape[-2], image.shape[-1])  # h, w
    #
    #     # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    #     # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    #     # Therefore it's important to use torch.Tensor.
    #     # dataset_dict["image"] = image
    #     # dataset_dict["label"] = label.to(torch.float)
    #     dataset_dict["depth"] = depth
    #     # dataset_dict["raw_image"] = raw_image
    #
    #     # Prepare per-category binary masks
    #     instances = Instances(image_shape)
    #     instances.gt_classes = classes
    #     if len(masks) == 0:
    #         # Some image does not have annotation (all ignored)
    #         instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
    #     else:
    #         masks = BitMasks(torch.stack(masks))
    #         instances.gt_masks = masks.tensor
    #
    #     dataset_dict["instances"] = instances
    #
    #     return dataset_dict