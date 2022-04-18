# Author: Aditya Saigal
# Define and configure the dataloader class to be used for training the FasterRCNN in Pytorch.

import json
import os
import torch
import matplotlib.image as mpimg
import torchvision
from torchvision import transforms
import numpy as np

# Following line of code allows us to run on Windows with a messed up OpenMP installation.
# Makes no difference for Linux.
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

# Give dataloader access to swap space so it doesn't run out of memory.
torch.multiprocessing.set_sharing_strategy('file_system')

class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, filename, target_categories, thermal = True):
        # Args:
            # root = root directory where data is stored
            # filename = coco file name containing annotations
            # target categories = dictionary mapping labels in FLIR dataset to our own custom labels
            # thermal  = bool variable used to denote that thermal images are being loaded
        
        # Need these class variables in subsequent functions
        self.thermal = thermal 
        self.root = root
        self.boxes, self.classes, self.files, self.ids = [], [], [], []
        # Temp variables storing data being processed
        boxes, classes, files = {}, {}, {}
        # Get data corresponding to each image
        with open(os.path.join(self.root, filename), "r") as file:
            data = json.load(file)
        # Iterate over each annotation in the file and build the targets for each image.
        for label in data["annotations"]:
            if not label["image_id"] in boxes:
                # This image has not yet been processed
                if(boxes):
                    if(not classes[curr_id]):
                        # The previous image contained none of the classes we want to detect (Pedestrain, Car and Bike)
                        # Thus we delete it from our dictionaries
                        del boxes[curr_id]
                        del classes[curr_id]
                        del files[curr_id]
                # Update the image id as we encounter a new image
                curr_id = label["image_id"]
                # Create lists storing bounding boxes and class labels for this image
                boxes[curr_id] = []
                classes[curr_id] = []

                # The image filename is stored in the images dictionary (in the json file)
                # Find the filename corresponding to the image id and add it to our list of files
                for image in data["images"]:
                    if(image["id"] == curr_id):
                        files[curr_id] = image["file_name"]

            if(label["category_id"] in target_categories):
                # As we iterate over class labels, only add a label if it is something we want to identify
                classes[curr_id].append(target_categories[label["category_id"]])
                boxes[curr_id].append([label["bbox"][0], label["bbox"][1], label["bbox"][0] + label["bbox"][2], label["bbox"][1] + label["bbox"][3]])

        # Unroll all info from the dictionaries into lists.
        # This is required as a Pytorch dataloader is referenced by index, not a key.
        for i, idx in enumerate(files):
            self.files.append(files[idx])
            self.classes.append(classes[idx])
            self.boxes.append(boxes[idx])
            self.ids.append(idx) # Assign the index of an image as its id.

    def __len__(self):
        # Return length of data list (need this by default)
        return len(self.files)

    def __getitem__(self, idx):
        # Allows the caller to index into the dataloader object -> makes this class essentially an iterator.

        # Given the index, get the corresponding data (image and labels).
        filename, boxes, classes = self.files[idx], torch.as_tensor(self.boxes[idx]), torch.as_tensor(self.classes[idx])

        # Read in the image as a numpy array.
        img_path = os.path.join(self.root, filename)
        img = mpimg.imread(img_path)

        # Extra processing to be done if the image is thermal (make it 3 channel)
        if(self.thermal):
            img = np.reshape(img, (1, img.shape[0], img.shape[1]))
            img_3_channel = np.concatenate((img, img, img), axis = 0)
            img = torch.as_tensor(img_3_channel).div(255)
        else:
            # If its rgb, just convert the image to a tensor
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        # Construct the ground truth output (target)
        target = {}
        target["boxes"] = boxes # location of bounding boxes
        target["labels"] = classes # class labels
        target["image_id"] = torch.tensor([self.ids[idx]]) # need the image id to compute stats
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # area corresponding to each bounding box (need this to get IOu stats)
        target["iscrowd"] = torch.zeros((len(classes),), dtype= torch.int64) #if iscrowd is set to one, the model ignores the input

        return img, target


if __name__ == "__main__":
    # Driver code to test the class implementation
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    target_categories = {1 : 1, 3 : 2, 12 : 3, 10 : 4}
    test = Image_Dataset("../images_thermal_train/", "coco.json", target_categories)

    image, target = test[2000]

    fig, ax = plt.subplots()
    ax.imshow(image[0])
    b_boxes = target["boxes"]
    for box in b_boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    print(target)
    plt.savefig("./test.png")
