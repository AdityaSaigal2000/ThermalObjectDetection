import torchvision
import torchvision.models as models
import torch.nn as nn
from torchvision.transforms import Resize
from dataloader import Image_Dataset
import torch
 
def collate_list(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    return data, target

def validate(model, val_loader, device):
    losses = []
    print("Running Validation")
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = model(images, targets)
            losses.append(loss)

        print("Class Loss: " + str(sum([d["loss_classifier"] for d in losses])/len(losses)))
        print("Loss Objectness: " + str(sum([d["loss_objectness"] for d in losses])/len(losses)))
        print("Loss Box Reg: " + str(sum([d["loss_box_reg"] for d in losses])/len(losses)))
        print("Loss RPN Box Reg: " + str(sum([d["loss_rpn_box_reg"] for d in losses])/len(losses)))

def train_rcnn(config):
    num_input_channels = config["num_input_channels"]
    target_categories = config["target_categories"]
    train_bs = config["train_bs"]
    val_bs = config["val_bs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    num_epochs = config["num_epochs"]

    backbone = models.mobilenet_v2(pretrained=True).features 
    backbone.out_channels = 1280
    anchor_generator = models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

    model = models.detection.FasterRCNN(backbone, num_classes = 3, rpn_anchor_generator = anchor_generator, box_roi_pool = roi_pooler)
    print(model)

    #model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    #model.roi_heads.box_predictor.cls_score = nn.Linear(1024, len(target_categories))
    #model.backbone.body.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size = (7, 7), stride = (2, 2), padding = {3, 3}, bias = False)
    
    print(model) 
    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        print("Using GPU")
        if(torch.cuda.device_count() > 1):
            torch.backends.cudnn.benchmark = True
            nn.DataParallel(model)
            print("Using " + str(torch.cuda.device_count()) + "GPUs")
    else:
        device = "cpu"

    print(device)
    model.to(device)


    train_set = Image_Dataset("../images_rgb_train/", "coco.json", target_categories)
    val_set = Image_Dataset("../images_rgb_val/", "coco.json", target_categories)
    train_loader =  torch.utils.data.DataLoader(train_set, batch_size = train_bs, shuffle = True, num_workers = 2, collate_fn = collate_list, drop_last = True)

    val_loader =  torch.utils.data.DataLoader(val_set, batch_size = val_bs, shuffle = True, num_workers = 4, collate_fn = collate_list, drop_last = True)
    
    optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size = 20, gamma = 0.5)
    model.train()
    for epoch in range(num_epochs): 
        epoch_losses = []
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
        
            epoch_losses.append(losses)
            loss = losses["loss_classifier"] + losses["loss_objectness"] + 4*losses["loss_box_reg"] + losses["loss_rpn_box_reg"]  
            optim.zero_grad()
            loss.backward()
            optim.step()
            
    
        scheduler.step()
        print("Epoch " + str(epoch))
        print("Class Loss: " + str(sum([d["loss_classifier"] for d in epoch_losses])/len(epoch_losses)))
        print("Loss Objectness: " + str(sum([d["loss_objectness"] for d in epoch_losses])/len(epoch_losses)))
        print("Loss Box Reg: " + str(sum([d["loss_box_reg"] for d in epoch_losses])/len(epoch_losses)))
        print("Loss RPN Box Reg: " + str(sum([d["loss_rpn_box_reg"] for d in epoch_losses])/len(epoch_losses)))

        #if(i%2):
        #    validate(model, val_loader, device)
    torch.save(model, "./faster_rcnn.pt")
    model.eval()
    for i in range(0, train_bs):
        input = torch.unsqueeze(images[i], dim = 0)
        output = model(input)
        idx = torchvision.ops.batched_nms(output[0]["boxes"], output[0]["scores"], output[0]["labels"], 0.6)
        
        img = torch.unsqueeze(images[i], dim = 0).cpu()[0]
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig, ax = plt.subplots()
        ax.imshow(img[0])
        for ix in idx:
            b_box = output[0]["boxes"][ix].cpu().detach().numpy()
            rect = patches.Rectangle((b_box[0], b_box[1]), b_box[2] - b_box[0], b_box[3] - b_box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.savefig("./test" + str(i) + ".png")
if(__name__ == "__main__"):
    training_config = {"num_input_channels" : 3, "target_categories" : {1 : 0, 2 : 1, 3 : 2},  "train_bs" : 20, "val_bs" : 10, "lr" : 1e-4, "weight_decay" : 0.1, "num_epochs" : 100}
    train_rcnn(training_config)


