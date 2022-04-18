# Author: Aditya Saigal
# Use the yolov5 API to make predictions using a custom trained model

import torch
from PIL import Image
import os

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path="runs/train/no_learn_upsample_yolo_x/weights/best.pt")

# Add all images to a list
images = []
for i, im in enumerate(os.listdir("/home/aditya/Capstone/no_learn_test/data")):
	images.append(Image.open(os.path.join("/home/aditya/Capstone/no_learn_test/data", im)))

# Perform inference on entire batch of images and output 512x512 images
results = model(images, size = 512)

# Print stats, such as inference time, precision and recall
results.print()
# Save outputs
results.save()
