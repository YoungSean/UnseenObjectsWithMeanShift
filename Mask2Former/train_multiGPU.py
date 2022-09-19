
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib', 'datasets'))

# ignore some warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
from detectron2.checkpoint import DetectionCheckpointer

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
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultTrainer
from train_net import Trainer


# import Mask2Former project
from mask2former import add_maskformer2_config
from tabletop_config import add_tabletop_config

# If we do not use detectron2 data mapper, set use_my_dataset as True
use_my_dataset = True
for d in ["train", "test"]:
    if use_my_dataset:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
    else:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))
    MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['__background__', 'object'])
metadata = MetadataCatalog.get("tabletop_object_train")

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg_file = "configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
#cfg_file = "configs/cityscapes/instance-segmentation/Base-Cityscapes-InstanceSegmentation.yaml"
cfg.merge_from_file(cfg_file)
add_tabletop_config(cfg)
if use_my_dataset:
    dataset = TableTopDataset(image_set="train", data_mapper=True)
    dataloader = build_detection_train_loader(cfg)

model = build_model(cfg)
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, optimizer, current_epoch=0):
    for batch, X in enumerate(dataloader):
        # Compute prediction and loss
        qualified_batch = []

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
def visualizeResult():
    im = cv2.imread("./rgb_00004.jpeg")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
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
