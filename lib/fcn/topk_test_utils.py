import sys
import os

from fcn.test_dataset import filter_labels_depth, crop_rois, clustering_features, match_label_crop

#print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Mask2Former'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
#print(os.path.join(os.path.dirname(__file__), '..', '..'))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.utils.visualizer import Visualizer
# from Mask2Former.mask2former import add_maskformer2_config
# from mask2former import add_maskformer2_config

from datasets import OCIDDataset
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys, os
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

from fcn.config import cfg
from fcn.test_common import _vis_minibatch_segmentation, _vis_features, _vis_minibatch_segmentation_final
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from utils.mean_shift import mean_shift_smart_init
from utils.evaluation import multilabel_metrics
import utils.mask as util_
from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from tabletop_config import add_tabletop_config
from torch.utils.data import DataLoader
# ignore some warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# build model
# cfg = get_cfg()
# add_deeplab_config(cfg)
# add_maskformer2_config(cfg)
# #cfg_file = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
# cfg_file = "../../Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
# cfg.merge_from_file(cfg_file)
# add_tabletop_config(cfg)
# # cfg.INPUT.INPUT_IMAGE = 'DEPTH'
# cfg.SOLVER.IMS_PER_BATCH = 1
# # cfg.MODEL.WEIGHTS = "/home/xy/yxl/UnseenObjectClusteringYXL/Mask2Former/output_RGB/model_0004999.pth"
# cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1


# dataset = TableTopDataset(data_mapper=True,eval=True)
# ocid_dataset = OCIDDataset(image_set="test")
#
# use_my_dataset = True
# #DatasetCatalog.register("tabletop_object_train", getTabletopDataset)
# for d in ["train", "test"]:
#     if use_my_dataset:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
#     else:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))
#     if cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 1:
#         MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['object'])
#     else:
#         MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['background', 'object'])
# metadata = MetadataCatalog.get("tabletop_object_train")

# Reference: https://www.reddit.com/r/computervision/comments/jb6b18/get_binary_mask_image_from_detectron2/

def get_confident_instances(outputs, topk=True, score=0.9, num_class=2, low_threshold=0.4):
    """
    Extract objects with high prediction scores.
    """
    instances = outputs["instances"]
    if topk:
        # we need to remove background predictions
        # keep only object class
        if num_class >= 2:
            instances = instances[instances.pred_classes == 1]
            confident_instances = instances[instances.scores > low_threshold]
            return confident_instances
        else:
            return instances
    confident_instances = instances[instances.scores > score]
    return confident_instances

def combine_masks(instances):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [2,3,0]. We assign labels from 2 since 1 stands for table.
    """
    mask = instances.get('pred_masks').to('cpu').numpy()
    num, h, w = mask.shape
    bin_mask = np.zeros((h, w))
    num_instance = len(mask)
    # if there is not any instance, just return a mask full of 0s.
    if num_instance == 0:
        return bin_mask

    for m, object_label in zip(mask, range(2, 2+num_instance)):
        label_pos = np.nonzero(m)
        bin_mask[label_pos] = object_label
    # filename = './bin_masks/001.png'
    # cv2.imwrite(filename, bin_mask)
    return bin_mask

class Predictor_RGBD(DefaultPredictor):

    def __call__(self, sample):
        """
        Args:
            sample: a dict of a data sample
            # ignore: original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            height, width = 480, 640
            original_image = cv2.imread(sample["file_name"])
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            transforms = self.aug.get_transform(original_image)
            image = transforms.apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            #image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}

            if self.cfg.INPUT.INPUT_IMAGE == "DEPTH" or "RGBD" in self.cfg.INPUT.INPUT_IMAGE:
                depth_image = sample["raw_depth"]
                depth_image = transforms.apply_image(depth_image)
                depth_image = torch.as_tensor(depth_image.astype("float32").transpose(2, 0, 1))
                depth = depth_image
                inputs["depth"] = depth

            predictions = self.model([inputs])[0]
            return predictions

class Network_RGBD(DefaultPredictor):

    def __call__(self, sample):
        """
        Args:
            sample: a dict of a data sample
            # ignore: original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            predictions = self.model([sample])[0]
            return predictions
def test_sample(cfg, sample, predictor, visualization = False, topk=True, confident_score=0.9, low_threshold=0.4):
    im = cv2.imread(sample["file_name"])
    print(sample["file_name"])
    if "label" in sample.keys():
        gt = sample["label"].squeeze().numpy()
    else:
        gt = sample["labels"].squeeze().numpy()

    # if cfg.INPUT.INPUT_IMAGE == "DEPTH":
    #     outputs = predictor(sample["raw_depth"])
    # else:
    #     outputs = predictor(im)
    image = sample['image_color'].cuda()
    sample["image"] = image
    sample["height"] = image.shape[-2]  # image: 3XHXW, tensor
    sample["width"] = image.shape[-1]
    outputs = predictor(sample)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score,
                                                  num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=low_threshold)
    binary_mask = combine_masks(confident_instances)
    # print(binary_mask)
    #np.save("pred51", binary_mask)
    metrics = multilabel_metrics(binary_mask, gt)
    print(f"metrics: ", metrics)
    ## Visualize the result
    if visualization:
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        visual_result = out.get_image()[:, :, ::-1]
        # cv2.imwrite(sample["file_name"][-6:-3]+"pred.png", visual_result)
        cv2.imshow("image", visual_result)
        cv2.waitKey(0)
        # cv2.waitKey(100000)
        cv2.destroyAllWindows()
    # markers = refine_with_watershed(im, binary_mask)
    # metrics2 = multilabel_metrics(markers, gt)
    # print(f"metrics2: ", metrics2g)
    return metrics

def refine_with_watershed(img, prediction):

    thresh = prediction.astype(int)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    opening = thresh
    opening = opening.astype('uint8')
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(sure_bg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    print(np.unique(markers))
    img[markers == -1] = [0, 0, 255]
    print("marker type: ", markers.dtype)
    markers = markers.astype(np.uint8)
    markers[markers==1] = 0
    markers[markers == 2] = 255
    markers[markers == 3] = 200
    markers[markers == 4] = 150
    markers[markers == 5] = 100
    cv.imshow("image marker", markers)
    # # print(np.unique(markers))
    cv2.waitKey(0)
    # #cv.imshow("image ", img)
    # #cv2.waitKey(0)
    # # cv2.waitKey(100000)
    cv2.destroyAllWindows()
    return markers

def get_result_from_network(cfg, image, depth, label, predictor, topk=True, confident_score=0.9, low_threshold=0.4, vis_crop=False):
    height = image.shape[-2]  # image: 3XHXW, tensor
    width = image.shape[-1]
    image = torch.squeeze(image, dim=0)
    depth = torch.squeeze(depth, dim=0)

    sample = {"image": image, "height": height, "width": width, "depth": depth}
    outputs = predictor(sample)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score,
                                                  num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=low_threshold)
    if vis_crop:
        im = image.cpu().numpy().transpose((1, 2, 0)) * 255.0
        im += np.array([[[102.9801, 115.9465, 122.7717]]])
        im = im.astype(np.uint8)
        cv2.imshow("image", im)
        cv2.waitKey(0)
        depth_blob = depth.cpu().numpy()
        depth = depth_blob[2]
        plt.imshow(depth)
        plt.axis('off')
        plt.show()

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        visual_result = out.get_image()[:, :, ::-1]
        # cv2.imwrite(sample["file_name"][-6:-3]+"pred.png", visual_result)
        cv2.imshow("image_segmentation", visual_result)
        cv2.waitKey(0)
        # cv2.waitKey(100000)
        cv2.destroyAllWindows()
    binary_mask = combine_masks(confident_instances)
    return binary_mask

def test_sample_crop(cfg, sample, predictor, predictor_crop, visualization = False, topk=True, confident_score=0.9, low_threshold=0.4, print_result=False):
    #cluster_crop = Predictor_RGBD_CROP(cfg)
    # First network: the image needs the original one.
    image = sample['image_color'].cuda() # for future crop
    sample["image"] = image
    sample["height"] = image.shape[-2] # image: 3XHXW, tensor
    sample["width"] = image.shape[-1]
    gt = None
    if "label" in sample.keys():
        gt = sample["label"].squeeze().numpy()
    elif "label" in sample.keys():
        gt = sample["labels"].squeeze().numpy()

    if gt is not None:
        label = torch.from_numpy(gt).unsqueeze(dim=0).cuda()
    depth = sample['depth'].cuda()

    outputs = predictor(sample)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score,
                                                  num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=low_threshold)
    binary_mask = combine_masks(confident_instances)
    metrics = multilabel_metrics(binary_mask, gt)
    if print_result:
        print("file name: ", sample["file_name"])
        print("first:", metrics)

    if visualization:
        im = cv2.imread(sample["file_name"])  # this is for visualization
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        visual_result = out.get_image()[:, :, ::-1]
        # cv2.imwrite(sample["file_name"][-6:-3]+"pred.png", visual_result)
        cv2.imshow("image", visual_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    out_label = torch.as_tensor(binary_mask).unsqueeze(dim=0).cuda()
    if len(depth.shape) == 3:
        depth = torch.unsqueeze(depth, dim=0)
    if len(image.shape) == 3:
        image = torch.unsqueeze(image, dim=0)
    if depth is not None:
        # filter labels on zero depth
        if 'OSD' in sample["file_name"]:
            out_label = filter_labels_depth(out_label, depth, 0.8)
        else:
            out_label = filter_labels_depth(out_label, depth, 0.5)



    # zoom in refinement
    out_label_refined = None
    if predictor_crop is not None:
        rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
        if rgb_crop.shape[0] > 0:
            labels_crop = torch.zeros((rgb_crop.shape[0], rgb_crop.shape[-2], rgb_crop.shape[-1]))#.cuda()
            for i in range(rgb_crop.shape[0]):
                binary_mask_crop = get_result_from_network(cfg, rgb_crop[i], depth_crop[i], out_label_crop[i], predictor_crop,
                                                       topk=topk, confident_score=confident_score, low_threshold=low_threshold)
                labels_crop[i] = torch.from_numpy(binary_mask_crop)
            out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)

    if visualization:
        bbox = None
        _vis_minibatch_segmentation_final(image, depth, label, out_label, out_label_refined, None,
            selected_pixels=None, bbox=bbox)

    if out_label_refined is not None:
        out_label_refined = out_label_refined.squeeze(dim=0).cpu().numpy()
    prediction = out_label.squeeze().detach().cpu().numpy()
    if out_label_refined is not None:
        prediction_refined = out_label_refined
    else:
        prediction_refined = prediction.copy()
    metrics_refined = multilabel_metrics(prediction_refined, gt)
    if print_result:
        print("refined: ", metrics_refined)
        print("========")

    return metrics, metrics_refined

def test_sample_crop_nolabel(cfg, sample, predictor, predictor_crop, visualization = False, topk=True, confident_score=0.9, low_threshold=0.4, print_result=False):
    image = sample['image_color'].cuda() # for future crop
    sample["image"] = image
    if len(image.shape) == 4:
        image = torch.squeeze(image, dim=0)
        print("image shape: ", image.shape)
    sample["height"] = image.shape[-2] # image: 3XHXW, tensor
    sample["width"] = image.shape[-1]

    depth = sample['depth'].cuda()
    if len(depth.shape) == 4:
        depth = torch.squeeze(depth, dim=0)

    outputs = predictor(sample)
    confident_instances = get_confident_instances(outputs, topk=topk, score=confident_score,
                                                  num_class=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                  low_threshold=low_threshold)
    binary_mask = combine_masks(confident_instances)

    if visualization:
        im = cv2.imread(sample["file_name"])  # this is for visualization
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(confident_instances.to("cpu"))
        visual_result = out.get_image()[:, :, ::-1]
        # cv2.imwrite(sample["file_name"][-6:-3]+"pred.png", visual_result)
        cv2.imshow("image", visual_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    out_label = torch.as_tensor(binary_mask).unsqueeze(dim=0).cuda()
    if len(depth.shape) == 3:
        depth = torch.unsqueeze(depth, dim=0)
    if len(image.shape) == 3:
        image = torch.unsqueeze(image, dim=0)
    if depth is not None:
        # filter labels on zero depth
        if 'OSD' in sample["file_name"]:
            out_label = filter_labels_depth(out_label, depth, 0.8)
        else:
            out_label = filter_labels_depth(out_label, depth, 0.5)

    # zoom in refinement
    out_label_refined = None
    if predictor_crop is not None:
        rgb_crop, out_label_crop, rois, depth_crop = crop_rois(image, out_label.clone(), depth)
        if rgb_crop.shape[0] > 0:
            labels_crop = torch.zeros((rgb_crop.shape[0], rgb_crop.shape[-2], rgb_crop.shape[-1]))#.cuda()
            for i in range(rgb_crop.shape[0]):
                binary_mask_crop = get_result_from_network(cfg, rgb_crop[i], depth_crop[i], out_label_crop[i], predictor_crop,
                                                       topk=topk, confident_score=confident_score, low_threshold=low_threshold)
                labels_crop[i] = torch.from_numpy(binary_mask_crop)
            out_label_refined, labels_crop = match_label_crop(out_label, labels_crop.cuda(), out_label_crop, rois, depth_crop)

    if visualization:
        bbox = None
        _vis_minibatch_segmentation_final(image, depth, None, out_label, out_label_refined, None,
            selected_pixels=None, bbox=bbox)

    if out_label_refined is not None:
        out_label_refined = out_label_refined.squeeze(dim=0).cpu().numpy()
    prediction = out_label.squeeze().detach().cpu().numpy()
    if out_label_refined is not None:
        prediction_refined = out_label_refined
    else:
        prediction_refined = prediction.copy()
    # metrics_refined = multilabel_metrics(prediction_refined, gt)
    # if print_result:
    #     print("refined: ", metrics_refined)
    #     print("========")

    # return metrics, metrics_refined
def test_dataset(cfg,dataset, predictor, visualization=False, topk=True, confident_score=0.9, low_threshold=0.4):
    metrics_all = []
    for i in trange(len(dataset)):
        metrics = test_sample(cfg, dataset[i], predictor, visualization=visualization,
                              topk=topk, confident_score=confident_score, low_threshold=low_threshold)
        metrics_all.append(metrics)
    # for i in tqdm(dataset):
    #     metrics = test_sample(i, predictor, visualization=visualization)
    #     metrics_all.append(metrics)
    print('========================================================')
    if not topk:
        print("Mask threshold: ", confident_score)
    else:
        print(f"We first pick top {cfg.TEST.DETECTIONS_PER_IMAGE} instances ")
        print(f"and get those whose class confidence > {low_threshold}.")

    print("weight: ", cfg.MODEL.WEIGHTS)
    result = {}
    num = len(metrics_all)
    print('%d images' % num)
    print('========================================================')
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]

    for k in sorted(result.keys()):
        result[k] /= num
        print('%s: %f' % (k, result[k]))

    print('%.6f' % (result['Objects Precision']))
    print('%.6f' % (result['Objects Recall']))
    print('%.6f' % (result['Objects F-measure']))
    print('%.6f' % (result['Boundary Precision']))
    print('%.6f' % (result['Boundary Recall']))
    print('%.6f' % (result['Boundary F-measure']))
    print('%.6f' % (result['obj_detected_075_percentage']))

    print('========================================================')
    print(result)
    print('====================END=================================')

def test_dataset_crop(cfg,dataset, predictor, network_crop, visualization=False, topk=True, confident_score=0.9, low_threshold=0.4):
    metrics_all = []
    metrics_all_refined = []
    for i in trange(len(dataset)):
        metrics, metrics_refined = test_sample_crop(cfg, dataset[i], predictor, network_crop, visualization=visualization,
                              topk=topk, confident_score=confident_score, low_threshold=low_threshold)
        metrics_all.append(metrics)
        metrics_all_refined.append(metrics_refined)
    # for i in tqdm(dataset):
    #     metrics = test_sample(i, predictor, visualization=visualization)
    #     metrics_all.append(metrics)
    print('========================================================')
    if not topk:
        print("Mask threshold: ", confident_score)
    else:
        print(f"We first pick top {cfg.TEST.DETECTIONS_PER_IMAGE} instances ")
        print(f"and get those whose class confidence > {low_threshold}.")

    print("weight: ", cfg.MODEL.WEIGHTS)
    result = {}
    num = len(metrics_all)
    print('%d images' % num)
    print('========================================================')
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]

    for k in sorted(result.keys()):
        result[k] /= num
        print('%s: %f' % (k, result[k]))

    print('%.6f' % (result['Objects Precision']))
    print('%.6f' % (result['Objects Recall']))
    print('%.6f' % (result['Objects F-measure']))
    print('%.6f' % (result['Boundary Precision']))
    print('%.6f' % (result['Boundary Recall']))
    print('%.6f' % (result['Boundary F-measure']))
    print('%.6f' % (result['obj_detected_075_percentage']))

    print('========================================================')
    print(result)
    print('====================Refined=============================')

    result_refined = {}
    for metrics in metrics_all_refined:
        for k in metrics.keys():
            result_refined[k] = result_refined.get(k, 0) + metrics[k]

    for k in sorted(result_refined.keys()):
        result_refined[k] /= num
        print('%s: %f' % (k, result_refined[k]))
    print(result_refined)
    print('========================================================')
def test_dataset_with_weight(weight_path, cfg, dataset,topk=True, confident_score=0.9):
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Predictor_RGBD(cfg)
    test_dataset(cfg, dataset, predictor, topk=topk, confident_score=confident_score)

def test_dataset_with_weight(weight_path, cfg, sample,topk=True, confident_score=0.9):
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Predictor_RGBD(cfg)
    test_dataset(cfg, sample, predictor, topk=topk, confident_score=confident_score)

#
# use_depth = True
# if use_depth:
#     cfg.INPUT.INPUT_IMAGE = 'DEPTH'
# # test_dataset(dataset, predictor)
#
# #weight_path = "../../Mask2Former/output_RGB_n2/model_final.pth"
# #cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
# weight_path = "../../Mask2Former/depth_output_n1_lr5/model_final.pth"
# cfg.MODEL.WEIGHTS = weight_path
# predictor = Predictor_RGBD(cfg)
# test_sample(ocid_dataset[4], predictor, visualization=True)
# # test_dataset(ocid_dataset, predictor, confident_score=0.9)
# #test_dataset(ocid_dataset, predictor)
# test_dataset(dataset, predictor)
# print(ocid_dataset[4])


