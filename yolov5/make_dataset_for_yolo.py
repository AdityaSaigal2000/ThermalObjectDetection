# Author: Aditya Saigal
# Given a directory containing images and annotations in coco format, populate another directory with data accepted by yolov5. Use symlinks to save space.

import os
import json


def gen_yolo_data(root, filename, target_dir, target_categories, train = False, test = True):
        # root = root directory containing data in coco format
        # filename = coco annotations file
        # target_dir = directory to populate
        # target_categories = dict of categories we want to identify
        # train and test flags are used to determine the subdirectories that need to be populated.

        if(train and test):
            raise Exception("Can't have both train and test flags set to true.")

        # Dictionaries to hold temp info. 
        # Boxes contains the bboxes for each image.
        # Classes contain class labels for each image.
        # Files contains the file name storing all images.
        # Dims contains the height and width of each image (need this for yolov5 format).
        boxes, classes, files, dims  = {}, {}, {}, {}

        # choose subdirectory based on mode
        if(train):
            dirs = "train"
        else:
            dirs = "val"

        if(test):
            dirs = "test"

        # Create directories if they don't exist
        if(not os.path.exists(target_dir)):
            os.mkdir(target_dir)
        
        if(not os.path.exists(os.path.join(target_dir, "images"))):
            os.mkdir(os.path.join(target_dir, "images"))
            os.mkdir(os.path.join(target_dir, "labels"))

        if(not os.path.exists(os.path.join(target_dir, "images", dirs))):
            os.mkdir(os.path.join(target_dir, "images", dirs))
            os.mkdir(os.path.join(target_dir, "labels", dirs))

        # Load the coco annotations file and iterate through it.
        with open(os.path.join(root, filename), "r") as file:
            data = json.load(file)

        for label in data["annotations"]:
            if not label["image_id"] in boxes:
                # New image to process.
                if(boxes):
                    if(not classes[curr_id]):
                        # Previous image did not contain any classes we want to predict over. Delete all attributes related to it.
                        del boxes[curr_id]
                        del classes[curr_id]
                        del files[curr_id]
                        del dims[curr_id]

                # Add attributes for new image in temporary lists
                curr_id = label["image_id"]
                boxes[curr_id] = []
                classes[curr_id] = []

                for image in data["images"]:
                    if(image["id"] == curr_id):
                        # Store the image's file name and dimensions
                        files[curr_id] = image["file_name"]
                        dims[curr_id] = [image["height"], image["width"]]

            # If the current annotation contains a class we want to predict, add it to the classes list.
            if(label["category_id"] in target_categories):
                classes[curr_id].append(target_categories[label["category_id"]])
                # Store bounding box attributes in coco format (center normalized + height and width normalized)
                x_center = (label["bbox"][0] + 0.5*label["bbox"][2])/dims[curr_id][1]
                y_center = (label["bbox"][1] + 0.5*label["bbox"][3])/dims[curr_id][0]
                width = label["bbox"][2]/dims[curr_id][1]
                height = label["bbox"][3]/dims[curr_id][0]
                # Add the bounding box to a target file.
                boxes[curr_id].append([str(x_center), str(y_center), str(width), str(height)])

        # Now construct symbolic link to original image and create a target (.txt) file for each image.       
        for i, idx in enumerate(files):
            # Create symlink in target directory
            sym_link = os.path.join(target_dir, "images", dirs, files[idx].split("/")[1])
            os.symlink(os.path.join(root, files[idx]), sym_link)

            # out_str contains the target info that will be written to the .txt file for this image
            out_str = ""

            # poplate out str.
            for i in range(len(classes[idx])):
                out_str += str(classes[idx][i]) + " " + " ".join(boxes[idx][i]) + "\n"
            
            # write out str to the required file and save
            with open(os.path.join(target_dir, "labels", dirs, files[idx].split("/")[1].split(".")[0] + ".txt"), "w+") as file:
                file.write(out_str)
            

if __name__ == "__main__":
    # Example usage: original files are stored at /home/aditya/Capstone/attn_upsample_test/ in coco format
    # Annotations file is coco.json
    # Want to populate ../datasets/attn_thermal/ with yolov5 formatted dataset
    gen_yolo_data("/home/aditya/Capstone/attn_upsample_test/", "coco.json", "../datasets/attn_thermal/", {1 : 0, 2 : 1, 3 : 2})
