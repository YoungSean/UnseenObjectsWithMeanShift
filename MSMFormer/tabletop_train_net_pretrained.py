# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import sys
import os
#print(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

# ignore some warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetMapper, DatasetCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from meanshiftformer import (
    UnseenInstanceDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)
from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from datasets.pushing_dataset import PushingDataset
from datasets.mixture_dataset import MixtureDataset
from datasets.uoais_dataset import UOAIS_Dataset
from tabletop_config import add_tabletop_config
from meanshiftformer.config import add_meanshiftformer_config

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "unseen_instance":
            mapper = UnseenInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            print("check the right data mapper!")

        dataloader = build_detection_train_loader(cfg,
                                                  mapper=DatasetMapper(cfg, is_train=True))
        return dataloader


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def run_step(self):
        self._trainer.iter = self.iter
        """
                Implement the AMP training logic.
                """
        assert self._trainer.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        qualified_batch = []

        for sample in data:
            if len(sample["instances"]) > 0:
                qualified_batch.append(sample)
        # print("after removing: ", len(qualified_batch))
        # skip empty batch
        if len(qualified_batch) == 0:
            return
        data = qualified_batch
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self._trainer.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self._trainer.optimizer.zero_grad()
        self._trainer.grad_scaler.scale(losses).backward()

        self._trainer._write_metrics(loss_dict, data_time)

        self._trainer.grad_scaler.step(self._trainer.optimizer)
        self._trainer.grad_scaler.update()

# some settings:

# If we do not use detectron2 data mapper, set use_my_dataset as True
use_my_dataset = True
for d in ["train", "test"]:
    if use_my_dataset:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
    else:
        DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))
    MetadataCatalog.get("tabletop_object_" + d).set(thing_classes=['__background__', 'object'])

    DatasetCatalog.register("pushing_object_" + d, lambda d=d: PushingDataset(d))
    MetadataCatalog.get("pushing_object_" + d).set(thing_classes=['__background__', 'object'])

    DatasetCatalog.register("mixture_object_" + d, lambda d=d: MixtureDataset(d))
    MetadataCatalog.get("mixture_object_" + d).set(thing_classes=['__background__', 'object'])

    DatasetCatalog.register("uoais_object_" + d, lambda d=d: UOAIS_Dataset(d))
    MetadataCatalog.get("uoais_object_" + d).set(thing_classes=['__background__', 'object'])
#
# metadata = MetadataCatalog.get("tabletop_object_train")



# metadata = MetadataCatalog.get("pushing_object_train")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    # cfg_file = "configs/UOAIS_ResNet50.yaml"
    cfg_file = "configs/UOAIS_UCN.yaml"
    cfg.merge_from_file(cfg_file)
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # some configs for demo training
    cfg.OUTPUT_DIR = "./RGB_uoais_0518_RGBD"
    # RGBD UCN
    # cfg.MODEL.WEIGHTS = "../data/checkpoints/output_1008_normal_BGR_model_0069999.pth"
    # COLOR, only RGB, UCN
    #cfg.MODEL.WEIGHTS = "../data/checkpoints/RGB_norm_model_0069999.pth"
    #cfg.MODEL.WEIGHTS = "./output_1229_Res50_learn_10dec/model_0017499.pth"
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="msmformer")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load()
    return trainer.train()




if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
