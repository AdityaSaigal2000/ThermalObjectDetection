# By: Raghav Srinivasan
# Training and evaluation code for FasterRCNN with detectron2 framework

from detectron2.data.datasets import register_coco_instances
import os
import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Register dataset:
register_coco_instances("thermal_train", {}, "./ThermalTrainSmall/thermal_train_small_coco_processed.json",  "./ThermalTrainSmall/")
register_coco_instances("thermal_val", {}, "./ThermalVal/thermal_val_coco_processed.json",  "./ThermalVal/")

# Create model:
cfg = get_cfg()
os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
cfg.merge_from_file("/home/aditya/Capstone2/ThermalObjectDetection/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("thermal_train")
cfg.DATASETS.TEST = ("thermal_train")

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = ("/home/aditya/Capstone2/ThermalObjectDetection/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

# Model hyperparameters:
cfg.INPUT.MAX_SIZE_TEST = 1300
cfg.INPUT.MAX_SIZE_TRAIN = 1300

cfg.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
cfg.SOLVER.BASE_LR = 1e-5
cfg.SOLVER.IMS_PER_BATCH = 21

cfg.SOLVER.STEPS = (1250, 1750, 2000, 2500, 3000)
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

cfg.SOLVER.CHECKPOINT_PERIOD = 5000

cfg.OUTPUT_DIR = "thermal_run2"
# Train:
trainer = DefaultTrainer(cfg)
trainer.train()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0004999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# Evaluate:
evaluator = COCOEvaluator("thermal_train", output_dir="./detectron_test_1")
val_loader = build_detection_test_loader(cfg, "thermal_train")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

