import detectron2

from datasets.tabletop_object import TableTopObject
from datasets.tabletop_dataset import TableTopObject2
from detectron2.data import MetadataCatalog

#dataset = TableTopObject2(image_set='train')
#print(dataset[2])
from detectron2.data import DatasetCatalog


def getTabletopDataset():
    dataset = TableTopObject2(image_set='train')
    print(len(dataset))
    dataset_dicts = []
    for i in range(len(dataset)):
        dataset_dicts.append(dataset[i])

    return dataset_dicts




DatasetCatalog.register("my_dataset", getTabletopDataset)
# later, to access the data:
data = DatasetCatalog.get("my_dataset")
print(len(data))


MetadataCatalog.get("my_dataset").thing_classes = ['__background__', 'object']

import random
import cv2
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt


metadata = MetadataCatalog.get("my_dataset")
# for d in random.sample(data, 10):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#
#     #plt.imshow(out.get_image())
#     window_name = 'image'
#     cv2.imshow(window_name, out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
#
# print("done!")

from detectron2.config import get_cfg
from detectron2 import model_zoo


def custom_config(num_classes=2):
    cfg = get_cfg()

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")
    # Input
    cfg.INPUT.MASK_FORMAT = "bitmask"  # alternative: "polygon"
    # Model
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64

    # Solver
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.STEPS = (20, 10000, 20000)
    cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 8


    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 20

    # INPUT
    #cfg.INPUT.MIN_SIZE_TRAIN = (480,)

    # DATASETS
    #cfg.DATASETS.TEST = ('val',)
    cfg.DATASETS.TRAIN = ('my_dataset',)

    # DATASETS
    cfg.OUTPUT_DIR = "./output_demo"

    return cfg


from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

if __name__ == '__main__':

    cfg = custom_config()

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()