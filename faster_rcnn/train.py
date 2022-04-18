# Author: Aditya Saigal
# Code to build and train FasterRCNN models

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torch.optim.lr_scheduler import StepLR
from image_dataloader import Image_Dataset

def build_model(backbone, num_classes, pretrained):
    # backbone is expected to be a string in the following list: ["ResNet50FPN", "MobileNetv2", "MobileNetv3FPN"]
    current_models = ["ResNet50FPN", "MobileNetv2", "MobileNetv3FPN"]
    if not backbone in current_models:
        raise Exception("Input backbone must be in the following list " + str(current_models))
    if backbone == "ResNet50FPN":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        num_classes = num_classes  + 1 # original classes + background

        # Create the box predictor head by using the output size from the backbone and the number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    
    elif backbone == "MobileNetv3FPN":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained)
        num_classes = num_classes  + 1 # original classes + background

        # Create the box predictor head by using the output size from the backbone and the number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    else:
        # MobileNetv2 FasterRCNN needs to be assembled using individual modules
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        # Create a MobileNetv2 backbone by using the feature layers from the network
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280 # Set the number of output channels to match the output from MobileNetv2

        # Create an anchor generator with options to create 15 different types of bounding boxes
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        
        # Create custom ROI pooling layer
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

        # Assemble model
        model = FasterRCNN(backbone,
                           num_classes = num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Constructed FasterRCNN with a " + backbone + " backbone. Model has " + str(total_params) + " trainable parameters.")
    return model


def get_transform(train):
    # Any preliminary transformations to input data are applied here.
    transforms = [T.ToTensor()]
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def train(seed, model, thermal, target_categories, train_data_root, val_data_root, batch_size, lr, num_epochs, lr_step_size, lr_gamma, model_save):
    # seed = random seed to be trained with
    # model = NN constructed from build_model function
    # thermal = True if we are dealing with thermal data
    # target_categories = dictionary converting teledyne FLIR labels to our custom labels
    # train_data_root = location of training data
    # val_data_root = location of validation data
    # batch_size + lr + num_epochs + lr_step_size + lr_gamma = hyperparameters for training
    # model_save = filename where trained model is saved

    # Use a GPU if there is one
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Setup training and validation datasets
    dataset =  Image_Dataset(train_data_root, "coco.json", target_categories, thermal = thermal)
    dataset_val = Image_Dataset(val_data_root, "coco.json", target_categories, thermal = thermal)
    #target_categories = {1 : 1, 2 : 2, 3 : 3}

    print(str(len(dataset)) + " Training Samples")
    print(str(len(dataset_val)) + " Validation Samples")
    # Set the random seed for training
    torch.manual_seed(seed)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = utils.collate_fn)
    data_loader_val = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = utils.collate_fn)
    # Send model to GPU
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr = lr) # Use Adam optimizer with custom learning rate

    # Use a step LR scheduler with custom hyperparameters.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = lr_step_size, gamma = lr_gamma)

    for epoch in range(num_epochs):
        # train for num_epochs and print stats at each epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        # update the learning rate using the lr_schedu;er
        lr_scheduler.step()
        # run model on val set
        evaluate(model, data_loader_val, device=device)

    torch.save(model, model_save)

if __name__ == "__main__":
    # build a pretrained model with a ResNet50FPN backbone for predicting 3 classes
    model = build_model("ResNet50FPN", 3, True)
    
    # Training Parameters
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

    # Now train the model
    train(seed, model, thermal, target_categories, train_data_root, val_data_root, batch_size, lr, num_epochs, lr_step_size, lr_gamma, model_save)

