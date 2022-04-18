# Originally from Pytorch Faster RCNN Tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
# Modified by: Aditya Saigal

import math
import sys
import time
import torch
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import json
import io
from contextlib import redirect_stdout

# Create a dictionary to collect training + validation stats at every epoch
training_data = {}
# These lists are filled with training data, which is used to populate the dictionary above. 
average_losses = []
average_val_losses = []
map_05_095 = []
map_05_095_large = []
map_05 = []
recall = []

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    all_losses = [] # Collected losses for each iteration in an epoch
    model.train()
    # Setting up the logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    # Removed warmup scheduler here. Slows down training for no reason

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Send inputs and targets to GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Get individual losses
        loss_dict = model(images, targets)
        # Add up the losses
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # Get the loss as a float value
        loss_value = losses_reduced.item()
        
        all_losses.append(loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Update network parameters
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # get average losses for epoch    
    average_losses.append(sum(all_losses)/len(all_losses))

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    all_losses = []
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()
        model.eval()
        with torch.no_grad():
            outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        model.train()
        with torch.no_grad():
            losses = model(images, targets)
        
            loss_dict_reduced = utils.reduce_dict(losses)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
    
            loss_value = losses_reduced.item()
            all_losses.append(loss_value)

        #model.eval()
        #with torch.no_grad():
        #    outputs = model(images)

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    average_val_losses.append(sum(all_losses)/len(all_losses)) 
    with io.StringIO() as buf, redirect_stdout(buf):
        coco_evaluator.summarize()
        output = buf.getvalue()
   
    
    # Take stdout from coco logger and place it in our lists for tracking purposes
    map_05_095.append(float(output.split("\n")[1].split("=")[-1]))
    map_05.append(float(output.split("\n")[2].split("=")[-1]))
    map_05_095_large.append(float(output.split("\n")[6].split("=")[-1]))
    recall.append(float(output.split("\n")[-2].split("=")[-1]))

    torch.set_num_threads(n_threads)
    
    # Populate dictionary and save in json format. To be parsed later for visualization.
    data["train"] = average_losses
    data["val"] = average_val_losses
    data["map_0.5_0.95"] = map_05_095
    data["map_0.5"]= map_05
    data["map_05_095_large"] = map_05_095_large
    data["recall"] = recall
    with open("rcnn_thermal_mb_3_data_lower_lr.json", "w+") as file:
        json.dump(data, file)

    return coco_evaluator
