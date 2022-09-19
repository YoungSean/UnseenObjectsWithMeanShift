# check some functions
import sys
import os
# print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib', 'datasets'))

# ignore some warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
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
from torch.utils.data import DataLoader
#from google.colab.patches import cv2_imshow
from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
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
from tabletop_config import add_tabletop_config
from torch.utils.data.dataloader import default_collate
#im = cv2.imread("./input.jpg")
use_my_dataset = True
#DatasetCatalog.register("tabletop_object_train", getTabletopDataset)
for d in ["train", "test"]:
    if use_my_dataset:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
    else:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))
    MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['__background__', 'object'])
metadata = MetadataCatalog.get("tabletop_object_train")
# later, to access the data:
#training_data = DatasetCatalog.get("tabletop_object_train")
# print(len(training_data))

# Show the image
# cv2.imshow("plane", im)
# cv2.waitKey(0)
# #closing all open windows
# cv2.destroyAllWindows()



cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg_file = "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
#cfg_file = "configs/cityscapes/instance-segmentation/Base-Cityscapes-InstanceSegmentation.yaml"
cfg.merge_from_file(cfg_file)
add_tabletop_config(cfg)
if use_my_dataset:
    dataset = TableTopDataset(image_set="train", data_mapper=True)#, data_mapper=DatasetMapper(cfg, is_train=True))
    dataloader = build_detection_train_loader(cfg)
#dataloader = build_detection_train_loader(DatasetRegistry.get("tabletop_object_train"), mapper=DatasetMapper(cfg, is_train=True))
# dataloader = DataLoader(dataset, batch_size=4)
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

model = build_model(cfg)
model.cuda()
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
times = 0
# for x in dataloader:
#     print(model(x))
#     times += 1
#     if times == 2:
#         break


def train_loop(dataloader, model, optimizer, current_epoch=0):
    for batch, X in enumerate(dataloader):
        # Compute prediction and loss

        ## Remove the batch containing some samples without objects
        # has_empty_instances = False
        # for sample in X:
        #     if len(sample["instances"]) == 0:
        #        has_empty_instances = True
        # if has_empty_instances:
        #     continue
        qualified_batch = []
        # print("before removing: ", len(X))

        for sample in X:
            if len(sample["instances"]) > 0:
               qualified_batch.append(sample)
        # print("after removing: ", len(qualified_batch))
        # skip empty batch
        if len(qualified_batch) == 0:
            continue
        X = qualified_batch
        loss_dict = model(X)
        detailed_loss = [(k, round(v.item(), 3)) for k,v in loss_dict.items()]
        losses = sum(loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if batch % 2 == 0:
            losses, current = losses.item(), batch * len(X)
            print(f"loss: {losses:.3f}  loss_dict: {detailed_loss}")

        if batch >= 10:
            print(batch)
            break


#sample = dataset[0]
train_loop(dataloader, model, optimizer)
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


# if __name__ == '__main__':
#     cfg.SOLVER.MAX_ITER = 2200 + 100
#     trainer = Trainer(cfg)
#     trainer.resume_or_load()
#     trainer.train()

    #cfg.MODEL.WEIGHTS = "./output/model_final.pth"

    #visualizeResult()





#closing all open windows



print("done!")
