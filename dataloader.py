import json
import os
import torch
import matplotlib.image as mpimg
import torchvision
from torchvision import transforms
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, filename, target_categories, thermal = True):
        # if thermal is true, then setup a dataloader with 3 channel thermal images (replicated)
        self.thermal = thermal
        boxes, classes, files = {}, {}, {}
        self.root = root
        self.boxes, self.classes, self.files = [], [], []

        with open(os.path.join(self.root, filename), "r") as file:
            data = json.load(file)
        for label in data["annotations"]:
            if not label["image_id"] in boxes:
                if(boxes):
                    if(not classes[curr_id]):
                        del boxes[curr_id]
                        del classes[curr_id]
                        del files[curr_id]
                curr_id = label["image_id"]

                boxes[curr_id] = []
                classes[curr_id] = []

                for image in data["images"]:
                    if(image["id"] == curr_id):
                        files[curr_id] = image["file_name"]

            if(label["category_id"] in target_categories):
                classes[curr_id].append(target_categories[label["category_id"]])

                boxes[curr_id].append([label["bbox"][0], label["bbox"][1], label["bbox"][0] + label["bbox"][2], label["bbox"][1] + label["bbox"][3]])



        for i, idx in enumerate(files):
            #if(i == 8000):
            #    break
            self.files.append(files[idx])
            self.classes.append(classes[idx])
            self.boxes.append(boxes[idx])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename, boxes, classes = self.files[idx], torch.as_tensor(self.boxes[idx]), torch.as_tensor(self.classes[idx])

        img_path = os.path.join(self.root, filename)
        img = mpimg.imread(img_path)
        if(self.thermal):
            img = np.reshape(img, (1, img.shape[0], img.shape[1]))
            img_3_channel = np.concatenate((img, img, img), axis = 0)
            img = torch.as_tensor(img_3_channel).div(255)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)

        #print(img.shape)
        #exit()
        #img = torch.as_tensor(torchvision.transforms.functional.to_tensor(img_3_channel.copy()))
        #print(img.shape)
        #exit()

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes

        return img, target

if __name__ == "__main__":
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
