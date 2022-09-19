# check some functions
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from lib.datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from detectron2.data import MetadataCatalog, DatasetMapper

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

# import some common libraries
import numpy as np
import cv2
import torch
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data import DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultTrainer
from train_net import Trainer
# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
#coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

# import Mask2Former project
from mask2former import add_maskformer2_config

#im = cv2.imread("./input.jpg")

#DatasetCatalog.register("tabletop_object_train", getTabletopDataset)
for d in ["train", "test"]:
    DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))
    MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['__background__', 'object'])
metadata = MetadataCatalog.get("tabletop_object_train")
# later, to access the data:
#training_data = DatasetCatalog.get("tabletop_object_train")
#print(len(training_data))

# Show the image
# cv2.imshow("plane", im)
# cv2.waitKey(0)
# #closing all open windows
# cv2.destroyAllWindows()

## set path
import sys
sys.path.append("../lib")
sys.path.append("../lib/datasets")

def add_tabletop_config(cfg):
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.INPUT.MASK_FORMAT = "bitmask"  # alternative: "polygon"
    cfg.MODEL.MASK_ON = True
    cfg.DATASETS.TRAIN = ("tabletop_object_train",)
    # cfg.DATASETS.TEST= ("tabletop_object_test",)
    cfg.DATASETS.TEST = ()
    cfg.INPUT.MIN_SIZE_TRAIN = (480,)
    cfg.INPUT.MIN_SIZE_TEST = (480,)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MAX_SIZE_TEST = 800
    cfg.SOLVER.MAX_ITER = 40
    #cfg.INPUT.CROP.ENABLED = False
    cfg.MODEL.WEIGHTS = "./output/model_final.pth"
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg_file = "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
#cfg_file = "configs/cityscapes/instance-segmentation/Base-Cityscapes-InstanceSegmentation.yaml"
cfg.merge_from_file(cfg_file)
add_tabletop_config(cfg)
# dataloader = build_detection_train_loader(cfg,
#    mapper=DatasetMapper(cfg, is_train=True))
# if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
#     mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
# elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
#     mapper = MaskFormerInstanceDatasetMapper(cfg, True)
# train_loader = build_detection_train_loader(cfg, mapper=mapper)
# cfg_file = "configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
# cfg.merge_from_file(cfg_file)
# cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
# cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
# cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
# cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
# model = build_model(cfg)
# predictor = DefaultPredictor(cfg)
# for x in dataloader:
#     #print(x)
#     outputs = model(x)
#     print(outputs)
#
# # Show panoptic/instance/semantic predictions:
# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
# panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
# instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
# v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
# semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
# print("Panoptic segmentation (top), instance segmentation (middle), semantic segmentation (bottom)")
# cv2.imshow("image", np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1])
#
#
# cv2.waitKey(0)
# # #closing all open windows
# cv2.destroyAllWindows()
def visualizeResult():
    im = cv2.imread("./rgb_00004.jpeg")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    # print(f"scores: ", outputs["instances"].scores)
    # print("result number: ", len(outputs["instances"].scores))
    # filter_result = outputs["instances"].scores > 0.2
    # print(filter_result)
    # indices = torch.nonzero(filter_result)
    # print(indices.shape)
    # indices = torch.squeeze(indices)
    # print(indices.shape)



    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    instances = outputs["instances"]
    confident_detections = instances[instances.scores > 0.3]

    out = v.draw_instance_predictions(confident_detections.to("cpu"))
    cv2.imshow("image", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    trainer = Trainer(cfg)
    trainer.resume_or_load()
    trainer.train()

    #cfg.MODEL.WEIGHTS = "./output/model_final.pth"

    #visualizeResult()





#closing all open windows



print("done!")
