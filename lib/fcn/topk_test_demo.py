import sys
import os



#print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'MSMFormer'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
#print(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.utils.visualizer import Visualizer
# from MSMFormer.mask2former import add_maskformer2_config
# from mask2former import add_maskformer2_config
from meanshiftformer.config import add_meanshiftformer_config
from datasets import OCIDDataset, OSDObject
from tqdm import tqdm, trange

from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from tabletop_config import add_tabletop_config
from torch.utils.data import DataLoader
# ignore some warnings
import warnings
import torch
from config import cfg
warnings.simplefilter("ignore", UserWarning)
from topk_test_utils import Predictor_RGBD, test_dataset, test_sample, test_sample_crop, test_dataset_crop, Network_RGBD
# get network crop
import networks
cfg.device = "cuda:0"
num_classes = 2
pretrained = "/home/xy/yxl/UnseenForMeanShift/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth"
pretrained_crop = "/home/xy/yxl/UnseenForMeanShift/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth"
network_name = "seg_resnet34_8s_embedding"

if pretrained_crop:
    network_data_crop = torch.load(pretrained_crop)
    network_crop = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda()
    network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda()
    network_crop.eval()
else:
    network_crop = None


def get_predictor(input_image="RGBD_ADD"):
# build model
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    #cfg_file = "/home/xy/yxl/UnseenObjectClusteringYXL/MSMFormer/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
    cfg_file = "../../MSMFormer/configs/tabletop_pretrained.yaml"
    cfg.merge_from_file(cfg_file)
    add_tabletop_config(cfg)
    cfg.SOLVER.IMS_PER_BATCH = 1 #
    # cfg.MODEL.WEIGHTS = "/home/xy/yxl/UnseenObjectClusteringYXL/MSMFormer/output_RGB/model_0004999.pth"
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 7
    cfg.INPUT.INPUT_IMAGE = input_image
    # arguments frequently tuned
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    weight_path = "../../MSMFormer/server_model/norm_model_0069999.pth"
    #cfg.device = "cuda:0"
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Network_RGBD(cfg)
    return predictor, cfg

def get_predictor_crop(input_image="RGBD_ADD"):
# build model
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    #cfg_file = "/home/xy/yxl/UnseenObjectClusteringYXL/MSMFormer/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
    #cfg_file = "../../MSMFormer/configs/crop_tabletop_pretrained.yaml"
    cfg_file = "../../MSMFormer/configs/crop_tabletop_pretrained.yaml"
    cfg.merge_from_file(cfg_file)
    add_tabletop_config(cfg)
    cfg.SOLVER.IMS_PER_BATCH = 1 #
    # cfg.MODEL.WEIGHTS = "/home/xy/yxl/UnseenObjectClusteringYXL/MSMFormer/output_RGB/model_0004999.pth"
    #cfg.MODEL.MASK_FORMER.DEC_LAYERS = 7
    cfg.INPUT.INPUT_IMAGE = input_image
    # arguments frequently tuned
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    weight_path = "../../MSMFormer/server_model/crop_dec9_model_final.pth"
    #cfg.device = "cuda:0"
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Network_RGBD(cfg)
    return predictor, cfg

predictor, cfg = get_predictor()
predictor_crop, cfg_crop = get_predictor_crop()

# cfg.INPUT.INPUT_IMAGE = 'RGBD_ADD' #"RGBD_ADD" #'DEPTH'

#weight_path = "../../MSMFormer/output_0923_kappa30/model_0139999.pth"


# test_dataset(dataset, predictor)


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

metrics, metrics_refined = test_sample_crop(cfg, ocid_dataset[10], predictor, predictor_crop, visualization=True, topk=False, confident_score=0.7, print_result=True)
test_sample_crop(cfg, osd_dataset[5], predictor, predictor_crop, visualization=True, topk=False, confident_score=0.7, print_result=True)

# met_all = []
# met_refined_all= []
# for i in range(400, 490, 8):
#     print(i)
#     metrics, metrics_refined = test_sample_crop(cfg, ocid_dataset[i], predictor, predictor_crop, visualization=True, topk=False, confident_score=0.7, print_result=True)
#     #metrics= test_sample(cfg, ocid_dataset[i], predictor, visualization=True, topk=True, confident_score=0.9)
#     met_all.append(metrics["Boundary F-measure"])
#     met_refined_all.append(metrics_refined["Boundary F-measure"])
# #
# print("Boundary F-measure", np.mean(np.array(met_all)))
# print("Refined Boundary F-measure", np.mean(np.array(met_refined_all)))

# OCID dataset
#test_dataset(cfg, ocid_dataset, predictor, visualization=False)
#test_dataset(cfg, ocid_dataset, predictor, visualization=True, topk=False, confident_score=0.9)
# test_dataset_crop(cfg, ocid_dataset, predictor, network_crop, visualization=False, topk=False, confident_score=0.9, num_of_ms_seed=5)
#test_dataset_crop(cfg, dataset, predictor, network_crop, visualization=False, topk=False, confident_score=0.9)

# test_dataset_crop(cfg, osd_dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)