import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

annot_file = "./images_thermal_train/coco.json"

with open(annot_file, "r") as file:
    data = json.load(file)

b_boxes = []
classes = []
for label in data["annotations"]:
    if(label["image_id"] == 20):
        b_boxes.append(label["bbox"])
        classes.append(label["category_id"])


for img in data["images"]:
    if(img["id"] == 20):
        image = mpimg.imread("./images_thermal_train/" + img["file_name"])
        fig, ax = plt.subplots()
        ax.imshow(image)
        for box in b_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.savefig("./test.png")


