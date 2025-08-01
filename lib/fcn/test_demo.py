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
from datasets.pushing_dataset import PushingDataset
from PIL import Image

from utils.evaluation import multilabel_metrics
# ignore some warnings
import warnings
import torch
from config import cfg
warnings.simplefilter("ignore", UserWarning)
from test_utils import test_dataset, test_sample, test_sample_crop, test_dataset_crop, Network_RGBD, test_sample_crop_nolabel, get_result_from_network

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

metadata = MetadataCatalog.get("tabletop_object_train")

mask_save_root = os.path.join('/media/gpuadmin/rcao/result/uois/ocid', 'msmformer_mask')

if __name__ == "__main__":
    # Here you can set the paths for networks
    # dirname = os.path.dirname(__file__)
    #
    # cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/configs/tabletop_pretrained.yaml')
    # weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/norm_model_0069999.pth")
    # cfg_file_MSMFormer_crop = os.path.join(dirname, "../../MSMFormer/configs/crop_tabletop_pretrained.yaml")
    # weight_path_MSMFormer_crop = os.path.join(dirname, "../../data/checkpoints/crop_dec9_model_final.pth")
    ocid_dataset = OCIDDataset(image_set="test")
    # osd_dataset = OSDObject(image_set="test")
    # pushing_dataset = PushingDataset("test")
    dataloader = torch.utils.data.DataLoader(ocid_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    predictor, cfg = get_predictor(cfg_file=cfg_file_MSMFormer,
                                   weight_path=weight_path_MSMFormer,
                                   input_image = "COLOR"
                                   )

    # predictor_crop, cfg_crop = get_predictor_crop(cfg_file=cfg_file_MSMFormer_crop,
    #                                               weight_path=weight_path_MSMFormer_crop)
    results =[]
    for sample_idx, sample in enumerate(dataloader):

        image_path = sample['filename'][0]
        image_name = os.path.basename(image_path).split('.')[0]
        image_dir = os.path.join(*os.path.dirname(image_path).split('/')[1:-1])
        
        # Example of predicting and visualizing samples from OCID and OSD dataset
        # pred_mask, metrics, metrics_refined = test_sample_crop(cfg, sample, predictor, None, visualization=False, topk=False, confident_score=0.7, print_result=True)
        # test_sample_crop(cfg, osd_dataset[5], predictor, predictor_crop, visualization=True, topk=False, confident_score=0.7, print_result=True)
        
        image = sample['image_color'].cuda() # for future crop
        pred_mask = get_result_from_network(cfg, image, None, None, predictor, False, 0.7, 0.4, False)
        gt_mask = sample["label"].squeeze().numpy()
        eval_metrics = multilabel_metrics(pred_mask.astype(np.uint8), gt_mask)
        print("file name: ", image_name)
        print("first:", eval_metrics)
        
        result = np.zeros(7)
        result[0] = eval_metrics['Objects F-measure']
        result[1] = eval_metrics['Objects Precision']
        result[2] = eval_metrics['Objects Recall']
        result[3] = eval_metrics['Boundary F-measure']
        result[4] = eval_metrics['Boundary Precision']
        result[5] = eval_metrics['Boundary Recall']
        result[6] = eval_metrics['obj_detected_075_percentage']
        results.append(result)
        
        print("Data type of pred_mask:", pred_mask.dtype)  # 打印数据类型
        print("Shape of pred_mask:", pred_mask.shape)      # 打印形状
        print("Minimum value in pred_mask:", np.min(pred_mask))  # 打印最小值
        print("Maximum value in pred_mask:", np.max(pred_mask))  # 打印最大值
        print("Unique values in pred_mask:", np.unique(pred_mask))  # 打印所有独特的值

        # pred_mask = (pred_mask / np.max(pred_mask)) * 255
        # result = Image.fromarray(pred_mask.astype(np.uint8))
        # mask_save_path = os.path.join(mask_save_root, image_dir)
        # os.makedirs(mask_save_path, exist_ok=True)
        # result.save(os.path.join(mask_save_path, '{}.png'.format(image_name)))

    results = np.stack(results, axis=0)
    print('Overlap Prec:{}, Rec:{}, F_score:{}, Boundary Prec:{}, Rec:{}, F_score:{}, %75:{}'. \
    format(np.mean(results[:, 1]), np.mean(results[:, 2]), np.mean(results[:, 0]),
            np.mean(results[:, 4]), np.mean(results[:, 5]), np.mean(results[:, 3]), np.mean(results[:, 6])))
    # np.save('OCID_msmformer_rgb_results_new.npy', results)
    
    # one stage model testing
    # test_dataset(cfg, pushing_dataset, predictor)
    # test_dataset(cfg, osd_dataset, predictor)
    # test_dataset(cfg, ocid_dataset, predictor)
    # test_sample(cfg, pushing_dataset[0], predictor, visualization=True)

    # Uncomment to predict a series of samples
    # met_all = []
    # met_refined_all= []
    # for i in range(1100, 1110,1):
    # # for i in [1560]:
    #     print(i)
    #     metrics, metrics_refined = test_sample_crop(cfg, ocid_dataset[i], predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)
    #     met_all.append(metrics["Boundary F-measure"])
    #     met_refined_all.append(metrics_refined["Boundary F-measure"])
    # print("Boundary F-measure", np.mean(np.array(met_all)))
    # print("Refined Boundary F-measure", np.mean(np.array(met_refined_all)))

    # # Uncomment to predict the whole dataset (OSD/OCID)
    # test_dataset_crop(cfg, ocid_dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)
    # test_dataset_crop(cfg, osd_dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)
    # test_dataset_crop(cfg, pushing_dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)
