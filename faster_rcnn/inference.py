# Author: Aditya Saigal
# Use a trained model to perform predictions on images from the FLIR dataset. Visualize predictions and save them for further use.

import os
import torch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

target_categories = {1 : 1, 2 : 2, 3 : 3} # Categories from FLIR dataset that we want to predict over.

# Setup dataset for testing.
dataset =  Image_Dataset("../../../Capstone/video_thermal_test/", "coco.json", target_categories, thermal = False)

# Use a GPU if possible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load trained model and set it to eval mode
model = torch.load("thermal_model.pt")
model.eval()

# Setup visualization parameters
colors = {"1" : "r", "2" : "b", "3" : "g"}
label = {"1" : "Person", "2" : "Bike", "3" : "Car"}
times = [] # List to get average inference times

# Iterate over entire dataset and perform predictions on each image. Visualize and save each prediction.

for i in range(len(dataset)):
    img, _ = dataset[i]
    with torch.no_grad():
        s = time.time()  
        # Make and time prediction      
        pred = model([img.to(device)])
        times.append(time.time() - s) # Save inference time

    # Perform NMS on the output, with an IOU threshold of 0.4. Function returns indices of all outputs that are relevant
    ix = torchvision.ops.batched_nms(pred[0]["boxes"], pred[0]["scores"], pred[0]["labels"], 0.4)

    # Get the image tensor as a numpy array anc convert to RGB format (for plotting)
    img = np.moveaxis(img.numpy(), 0, -1)

    # Plot the image and predicted outputs
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Loop over relevant outputs and plot them
    for idx in ix:
        # Get bounding box, confidence score and predicted labels
        b_box = pred[0]["boxes"][idx].cpu().detach().numpy()
        score = str(round(pred[0]["scores"][idx].cpu().detach().item(), 2))
        
        # Set custom color for the specific class label
        color = colors[str(pred[0]["labels"][idx].item())]
        # Bounding box rectangle
        rect = patches.Rectangle((b_box[0], b_box[1]), b_box[2] - b_box[0], b_box[3] - b_box[1], linewidth=1, edgecolor=color, facecolor='none')
        
        # Rectangle on top of bounding box to display class label and confidence score
        rect2 = patches.Rectangle((b_box[0], b_box[1] - 15), 70, 15, linewidth = 1, edgecolor = color, facecolor = color) 
        # Add rectangle to image
        ax.add_patch(rect)
        ax.add_patch(rect2)
        rx, ry = rect2.get_xy()

        # Get center of rect2
        cx = rx + rect2.get_width()/2.0
        cy = ry + rect2.get_height()/2.0

        # Annotate rect2
        ax.annotate(label[str(pred[0]["labels"][idx].item())] + " "  + score, (cx, cy), color = 'w', fontsize = 7, ha = 'center', va = 'center')

    # save plot
    if(not os.path.exists("./test_thermal/")):
        os.mkdir("./test_thermal/")
    plt.savefig("./test_thermal/img" + str(i) + ".png")
    plt.close()


print("Average Inference Time: " + str(sum(times)/len(times)))

