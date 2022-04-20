# Author: Raghav Srinivasan
# This file creates a folder for which to symlink dataset images, utilizing a processed COCO json

import os
import json

def symlink_data(current_dir, linked_dir, json_name):
	with open(os.path.join(current_dir, json_name), "r") as file:
	    data = json.load(file)

	for image in data["images"]:
		filename = image["file_name"]
		os.symlink(os.path.join(linked_dir, filename), os.path.join(current_dir, filename))

# symlink_data("ThermalObjectDetection/ThermalTrain", "images_thermal_train", "thermal_train_coco_processed.json")
# symlink_data("ThermalObjectDetection/ThermalVal", "images_thermal_val", "thermal_val_coco_processed.json")
# symlink_data("ThermalObjectDetection/ThermalTest", "video_thermal_test", "thermal_test_coco_processed.json")	
symlink_data("ThermalObjectDetection/ThermalTrainSmall", "images_thermal_train", "thermal_train_small_coco_processed.json")