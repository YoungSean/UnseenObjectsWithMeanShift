import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'MSMFormer'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

import numpy as np
import cv2
from tqdm import tqdm
from detectron2.data import MetadataCatalog, DatasetCatalog
from meanshiftformer.config import add_meanshiftformer_config
from datasets import OCIDDataset, OSDObject
from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from tabletop_config import add_tabletop_config
from datasets.pushing_dataset import PushingDataset
from PIL import Image

from utils.evaluation import multilabel_metrics
# ignore some warnings
import warnings
import torch
from config import cfg
warnings.simplefilter("ignore", UserWarning)
from test_utils import test_dataset, test_sample, test_sample_crop, test_dataset_crop, Network_RGBD, test_sample_crop_nolabel, get_result_from_network
from utils.mask import visualize_segmentation
dirname = os.path.dirname(__file__)


# # RGB
cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/configs/tabletop_pretrained_ResNet50.yaml')
weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/tabletop_rgb/norm_RGB_pretrained.pth") 
# weight_path_MSMFormer = os.path.join(dirname, "../../MSMFormer/norm_0111_RGB_mixture2_updated/model_0000319.pth") 

# # RGBD
# cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/configs/mixture_UCN.yaml')
# weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/rgbd_finetuned/norm_RGBD_finetuned_data04_OCID_5epoch.pth")

# cfg_file_MSMFormer_crop = os.path.join(dirname, "../../MSMFormer/configs/crop_mixture_UCN.yaml")
# weight_path_MSMFormer_crop = os.path.join(dirname, "../../data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth")

def get_general_predictor(cfg_file, weight_path, input_image="RGBD_ADD"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    cfg_file = cfg_file
    cfg.merge_from_file(cfg_file)
    add_tabletop_config(cfg)
    cfg.SOLVER.IMS_PER_BATCH = 1  #

    cfg.INPUT.INPUT_IMAGE = input_image
    if input_image == "RGBD_ADD":
        cfg.MODEL.USE_DEPTH = True
    else:
        cfg.MODEL.USE_DEPTH = False
    # arguments frequently tuned
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    weight_path = weight_path
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Network_RGBD(cfg)
    return predictor, cfg
def get_predictor(cfg_file=cfg_file_MSMFormer, weight_path=weight_path_MSMFormer, input_image="RGBD_ADD"):
    return get_general_predictor(cfg_file, weight_path, input_image=input_image)

# def get_predictor_crop(cfg_file=cfg_file_MSMFormer_crop, weight_path=weight_path_MSMFormer_crop, input_image="RGBD_ADD"):
#     return get_general_predictor(cfg_file, weight_path, input_image=input_image)

# set datasets
# use_my_dataset = True
# for d in ["train", "test"]:
#     if use_my_dataset:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
#     else:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))

def process_label(foreground_labels):
    """ Process foreground_labels
            - Map the foreground_labels to {0, 1, ..., K-1}

        @param foreground_labels: a [H x W] numpy array of labels

        @return: foreground_labels
    """
    # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
    unique_nonnegative_indices = np.unique(foreground_labels)
    mapped_labels = foreground_labels.copy()
    for k in range(unique_nonnegative_indices.shape[0]):
        mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
    foreground_labels = mapped_labels
    return foreground_labels

def read_file(file_path):
    f = open(file_path,"r")
    lines = f.readlines()
    data_list = []
    for line in lines:
        data_list.append(line.strip('\n'))
    return data_list

metadata = MetadataCatalog.get("tabletop_object_train")

dataset = 'PhoCAL' # 'OCID' 'PhoCAL' 'OSD'
dataset_root = os.path.join("/media/gpuadmin/rcao/dataset", dataset)
mask_save_root = os.path.join('/media/gpuadmin/rcao/result/uois', dataset, 'msmformer_mask')
os.makedirs(mask_save_root, exist_ok=True)
vis_save = True
vis_save_root = os.path.join(mask_save_root, 'vis')
os.makedirs(vis_save_root, exist_ok=True)
image_list = read_file(os.path.join(dataset_root, 'data_list.txt'))

predictor, cfg = get_predictor(cfg_file=cfg_file_MSMFormer,
                                weight_path=weight_path_MSMFormer,
                                input_image = "COLOR"
                                )
# load image
results =[]
for image_idx, image_path in enumerate(tqdm(image_list)):
    image_name = os.path.basename(image_path).split('.')[0]

    if dataset == 'OCID':
        image_dir = os.path.join(*os.path.dirname(image_path).split('/')[:-1])
        gt_mask_path = os.path.join(dataset_root, image_path.replace('rgb', 'label'))
        gt_mask = np.array(Image.open(gt_mask_path))
        gt_mask[gt_mask == 1] = 0
        if 'table' in gt_mask_path:
            gt_mask[gt_mask == 2] = 0

    elif dataset == 'OSD':
        image_dir = ''
        gt_mask_path = os.path.join(dataset_root, image_path.replace('image_color', 'annotation'))
        gt_mask = np.array(Image.open(gt_mask_path))

    elif dataset == 'PhoCAL':
        image_dir = os.path.join(*os.path.dirname(image_path).split('/')[:-1])
        gt_mask_path = os.path.join(dataset_root, image_path.replace('rgb', 'mask'))
        gt_mask = np.array(Image.open(gt_mask_path))
        
    im = cv2.imread(os.path.join(dataset_root, image_path))
    image = (torch.from_numpy(im).permute(2, 0, 1) - torch.Tensor([123.675, 116.280, 103.530]).view(-1, 1, 1).float()) / torch.Tensor([58.395, 57.120, 57.375]).view(-1, 1, 1).float()
    
    # gt_mask = process_label(gt_mask)

    pred_mask = get_result_from_network(cfg, image, None, None, predictor, False, 0.7, 0.4, False)
    # eval_metrics = multilabel_metrics(pred_mask.astype(np.uint8), gt_mask)
    # print("file name: ", image_name)
    # print("first:", eval_metrics)
    
    # result = np.zeros(7)
    # result[0] = eval_metrics['Objects F-measure']
    # result[1] = eval_metrics['Objects Precision']
    # result[2] = eval_metrics['Objects Recall']
    # result[3] = eval_metrics['Boundary F-measure']
    # result[4] = eval_metrics['Boundary Precision']
    # result[5] = eval_metrics['Boundary Recall']
    # result[6] = eval_metrics['obj_detected_075_percentage']
    # results.append(result)
    
    # print("Data type of pred_mask:", pred_mask.dtype)  # 打印数据类型
    # print("Shape of pred_mask:", pred_mask.shape)      # 打印形状
    # print("Minimum value in pred_mask:", np.min(pred_mask))  # 打印最小值
    # print("Maximum value in pred_mask:", np.max(pred_mask))  # 打印最大值
    # print("Unique values in pred_mask:", np.unique(pred_mask))  # 打印所有独特的值

    # pred_mask = (pred_mask / np.max(pred_mask)) * 255
    # result = Image.fromarray(pred_mask.astype(np.uint8))
    # mask_save_path = os.path.join(mask_save_root, image_dir)
    # os.makedirs(mask_save_path, exist_ok=True)
    # result.save(os.path.join(mask_save_path, '{}.png'.format(image_name)))

    if vis_save:
        visualize_segmentation(im=im, masks=pred_mask, save_dir=os.path.join(vis_save_root, "{}_seg.png".format(image_name)))
# results = np.stack(results, axis=0)
# print('Overlap Prec:{}, Rec:{}, F_score:{}, Boundary Prec:{}, Rec:{}, F_score:{}, %75:{}'. \
# format(np.mean(results[:, 1]), np.mean(results[:, 2]), np.mean(results[:, 0]),
#         np.mean(results[:, 4]), np.mean(results[:, 5]), np.mean(results[:, 3]), np.mean(results[:, 6])))
# np.save('OCID_msmformer_rgb_results_new.npy', results)
