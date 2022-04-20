# Author: Raghav Srinivasan
# Sample points from a dataset, to create a smaller dataset (to overfit to test training functionality)

import json
import os
import torch
import matplotlib.image as mpimg
import torchvision
from torchvision import transforms
import numpy as np
import random

path_to_json = 'thermal_train_coco_processed.json'
with open(path_to_json, "r") as file:
	data = json.load(file)

# New dictionary we wish to fill:
new_data = data.copy()

random.seed(42)
num_images_to_keep = 200
random_images_by_index = random.sample(range(0, len(data['images'])), num_images_to_keep)

images_to_delete = list(set(range(0, len(data['images']))) - set(random_images_by_index))
images_to_delete = sorted(images_to_delete, reverse=True)

for idx in images_to_delete:
	del new_data['images'][idx]

# Convert image ids of new image list to their index:
# All images:
image_id_to_index = {}
for i in range(len(new_data['images'])):
	image = new_data['images'][i]
	image_id_to_index[image['id']] = i

# Delete annotations if their image id is not in this list
for i in range(len(data['annotations'])-1, -1, -1):
	annotation = data['annotations'][i]
	if (not annotation['image_id'] in image_id_to_index):
		del new_data['annotations'][i]

with open(os.path.join(os.path.dirname(path_to_json), 'thermal_train_small_coco_processed.json'), 'w') as fp:
    json.dump(new_data, fp)
