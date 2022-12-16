import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'MSMFormer'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from meanshiftformer.config import add_meanshiftformer_config
from datasets import OCIDDataset, OSDObject
from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from tabletop_config import add_tabletop_config

# ignore some warnings
import warnings
import torch
from config import cfg
warnings.simplefilter("ignore", UserWarning)
from test_utils import test_dataset, test_sample, test_sample_crop, test_dataset_crop, Network_RGBD

dirname = os.path.dirname(__file__)

cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/configs/tabletop_pretrained.yaml')
weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/norm_model_0069999.pth")  # norm_model_0069999.pth

cfg_file_MSMFormer_crop = os.path.join(dirname, "../../MSMFormer/configs/crop_tabletop_pretrained.yaml")
weight_path_MSMFormer_crop = os.path.join(dirname, "../../data/checkpoints/crop_dec9_model_final.pth") # "../../MSMFormer/server_model/crop_dec9_model_final.pth" # crop_dec9_model_final.pth

def get_general_predictor(cfg_file, weight_path, input_image="RGBD_ADD"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    cfg_file = cfg_file
    cfg.merge_from_file(cfg_file)
    add_tabletop_config(cfg)
    cfg.SOLVER.IMS_PER_BATCH = 1  #

    cfg.INPUT.INPUT_IMAGE = input_image
    # arguments frequently tuned
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    weight_path = weight_path
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Network_RGBD(cfg)
    return predictor, cfg
def get_predictor(cfg_file=cfg_file_MSMFormer, weight_path=weight_path_MSMFormer, input_image="RGBD_ADD"):
    return get_general_predictor(cfg_file, weight_path, input_image=input_image)

def get_predictor_crop(cfg_file=cfg_file_MSMFormer_crop, weight_path=weight_path_MSMFormer_crop, input_image="RGBD_ADD"):
    return get_general_predictor(cfg_file, weight_path, input_image=input_image)

# set datasets
#dataset = TableTopDataset(data_mapper=True,eval=True)
ocid_dataset = OCIDDataset(image_set="test")
osd_dataset = OSDObject(image_set="test")

use_my_dataset = True
for d in ["train", "test"]:
    if use_my_dataset:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
    else:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))

metadata = MetadataCatalog.get("tabletop_object_train")

if __name__ == "__main__":
    # Here you can set the paths for networks
    # dirname = os.path.dirname(__file__)
    #
    # cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/configs/tabletop_pretrained.yaml')
    # weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/norm_model_0069999.pth")
    # cfg_file_MSMFormer_crop = os.path.join(dirname, "../../MSMFormer/configs/crop_tabletop_pretrained.yaml")
    # weight_path_MSMFormer_crop = os.path.join(dirname, "../../data/checkpoints/crop_dec9_model_final.pth")

    predictor, cfg = get_predictor(cfg_file=cfg_file_MSMFormer,
                                   weight_path=weight_path_MSMFormer)
    predictor_crop, cfg_crop = get_predictor_crop(cfg_file=cfg_file_MSMFormer_crop,
                                                  weight_path=weight_path_MSMFormer_crop)
    # Example of predicting and visualizing samples from OCID and OSD dataset
    metrics, metrics_refined = test_sample_crop(cfg, ocid_dataset[10], predictor, predictor_crop, visualization=True, topk=False, confident_score=0.7, print_result=True)
    test_sample_crop(cfg, osd_dataset[5], predictor, predictor_crop, visualization=True, topk=False, confident_score=0.7, print_result=True)

    # Uncomment to predict a series of samples
    # met_all = []
    # met_refined_all= []
    # for i in range(400, 490, 8):
    #     print(i)
    #     metrics, metrics_refined = test_sample_crop(cfg, ocid_dataset[i], predictor, predictor_crop, visualization=False, topk=False, confident_score=0.9, print_result=True)
    #     met_all.append(metrics["Boundary F-measure"])
    #     met_refined_all.append(metrics_refined["Boundary F-measure"])
    # print("Boundary F-measure", np.mean(np.array(met_all)))
    # print("Refined Boundary F-measure", np.mean(np.array(met_refined_all)))

    # Uncomment to predict the whole dataset (OSD/OCID)
    # test_dataset_crop(cfg, osd_dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)
    # test_dataset_crop(cfg, ocid_dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)