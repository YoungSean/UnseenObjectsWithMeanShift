def add_tabletop_config(cfg):
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.SOLVER.IMS_PER_BATCH = 1 #8,64,16
    cfg.INPUT.MASK_FORMAT = "bitmask"  # alternative: "polygon"
    cfg.MODEL.MASK_ON = True
    cfg.DATASETS.TRAIN = ("tabletop_object_train",)
    #cfg.MODEL.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    #cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    #cfg.MODEL.PIXEL_STD = [0, 0, 0]
    # cfg.DATASETS.TEST= ("tabletop_object_test",)
    cfg.DATASETS.TEST = ()
    cfg.INPUT.MIN_SIZE_TRAIN = (480,)
    #cfg.INPUT.MIN_SIZE_TEST = (480,)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MAX_SIZE_TEST = 800
    #cfg.SOLVER.MAX_ITER = 280000
    #cfg.SOLVER.CHECKPOINT_PERIOD = 17500
    #cfg.SOLVER.STEPS = (2500,) # (17500, 35000)
    #cfg.INPUT.CROP.ENABLED = False
    # cfg.MODEL.WEIGHTS = "./output/model_final.pth"
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True

    # Some configs to be modified
    cfg.DATALOADER.NUM_WORKERS = 6

    # set input data mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_instance"
    cfg.INPUT.RANDOM_FLIP = "none" # no flip for default dataloader
    # Same image size
    cfg.CUDNN_BENCHMARK = True
    # set output dir
    # cfg.OUTPUT_DIR = "./output"
    cfg.INPUT.INPUT_IMAGE = 'RGB'
    # no evaluation during training
    cfg.TEST.EVAL_PERIOD = 0
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2

    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    #
    cfg.SOLVER.BASE_LR = 0.0001