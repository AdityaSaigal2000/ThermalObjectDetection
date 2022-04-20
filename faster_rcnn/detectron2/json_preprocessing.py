# Author: Raghav Srinivasan
# Prune dataset (in coco format) to remove all labels that are not person, bike, or car.
# Also remove all images that have all their labels removed

import json
import os
import torch
import matplotlib.image as mpimg
import torchvision
from torchvision import transforms
import numpy as np

path_to_json = "thermal_test_coco.json"
with open(path_to_json, "r") as file:
	data = json.load(file)

# New dictionary we wish to fill:
new_data = data.copy()
# Prune all categories we don't care about:
new_data['categories'] = data['categories'][:3]
# print(new_data['categories'])

images_to_keep = []
annotations_to_remove = []
# Prune in reverse order, since we delete data by index:
# Going in reverse order does not affect index values of later items we wish to delete
for i in range(len(data['annotations'])-1, -1, -1):
	annotation = data['annotations'][i]
	if (annotation['category_id'] == 1 or annotation['category_id'] == 2 or annotation['category_id'] == 3):
		images_to_keep  += [annotation['image_id']]
	else:
		del new_data['annotations'][i]

# All images:
image_id_to_index = {}
for i in range(len(data['images'])):
	image = data['images'][i]
	image_id_to_index[image['id']] = i

images_to_delete = list(set(image_id_to_index.keys()) - set(images_to_keep))
images_to_delete_by_index = []
for image_id in images_to_delete:
	images_to_delete_by_index += [image_id_to_index[image_id]]

images_to_delete_by_index = sorted(images_to_delete_by_index, reverse=True)
for idx in images_to_delete_by_index:
	del new_data['images'][idx]

# Save as json:
path_to_new_json = "thermal_test_coco_processed.json"
with open(path_to_new_json, 'w') as fp:
    json.dump(new_data, fp)