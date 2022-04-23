# ThermalObjectDetection

Relying solely on RGB Images is unsafe for Object Detection with Autonomous Vehicles. These images provide no information in poor lighting conditions. 

![alt text](https://i.imgur.com/uobK6So.jpg)
![alt text](https://i.imgur.com/tZUKitz.jpeg)

We explore Object Detection on Thermal Images using the FLIR dataset, with models from YOLOv5 (Ultralytics) and FasterRCNN (Pytorch). As can be seen, thermal data can provide more information when RGB fails.

To run this project, clone this repository and download the Teledyne FLIR Dataset from [the official website](https://www.flir.ca/oem/adas/adas-dataset-form/).

The dataset comes with train, test and validation data (each with a coco.json file).

**To train or perform inference you'll need access to at least one RTX 3080 TI GPU (or preferably and RTX5000)**

## Training the FasterRCNN
> A custom FasterRCNN can be trained from scratch by modifying the following lines in __main__, in train.py:

    model = build_model("ResNet50FPN", 3, True)
    seed = 2022
    thermal = True
    target_categories = {1 : 1, 2 : 2, 3 : 3}
    train_data_root = "../images_thermal_train/"
    val_data_root = "../images_thermal_val/"
    batch_size = 40
    lr = 1e-5
    num_epochs = 15
    lr_step_size = 3
    lr_gamma = 0.7
    model_save = "first_model.pt"
    
The arguments to build_model are: Backbone (currently supporting ResNet50FPN, MobileNetv2 and MobileNetv3FPN), num_classes and thermal (true when training over thermal data).

target_categories is a dictionary mapping label indices in the FLIR dataset to our own set of indices. For example, to train a model detecting Street Sign (10), Skateboard (12) and Scooter (14), target_categories is : {10 : 1, 12: 2, 14: 3}.

## Inference with the FasterRCNN
> To produce output images, modify the following lines in inference.py:

    target_categories = {1 : 1, 2 : 2, 3 : 3} 
    dataset =  Image_Dataset("../../../Capstone/video_thermal_test/", "coco.json", target_categories, thermal = False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load("thermal_model.pt")  
    model.eval()
    colors = {"1" : "r", "2" : "b", "3" : "g"}
    label = {"1" : "Person", "2" : "Bike", "3" : "Car"}
    
dataset now takes in the path to the testing data and model refers to the trained model created by test.py. The colors dictionary is used to create a bounding box color for each class and label maps the class index to the class name.

## Training YOLO

We use the implementation of YOLO provided by Ultralytics (YOLOv5). To convert the FLIR data (using the COCO format) to the YOLOv5 format, modify the following line of code in make_dataset_for_yolo.py:

   ```gen_yolo_data("/home/aditya/Capstone/attn_upsample_test/", "coco.json", "../datasets/attn_thermal/", {1 : 0, 2 : 1, 3 : 2})```
   
The first input is the root directory containing FLIR data, followed by the name of the coco annotations file and the directory where the data for YOLO must be created.

With the data ready, create a YAML file, specifying configuration of your input (similar to thermal_dataset.yaml):
  ```path: ../datasets/upsampled_thermal
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  images/test # test images (optional)

# Classes
nc: 3  # number of classes
ch: 1 # number of input channels
names: ['person', 'bike', 'car']
```

A customized YOLO implementation can now be trained from terminal using a command similar to:
```python train.py --img 640 --batch 21 --epochs 60 --data thermal_dataset.yaml --weights yolov5s.pt --device 0,1,2```

See the [Ultralytics Documentation](https://github.com/ultralytics/yolov5/) for more information.

## Inference with YOLO
To construct images with bounding boxes, based on the model output, use test.py and update the path to the model and data directories.

## Making a Video of the Output Images (FasterRCNN and YOLO)
Once all images with model predictions are in a single directory, use make_video.py to create a video from the images at 24fps.
Update the img_folder and the video_name to get your customized images in a customized video.


