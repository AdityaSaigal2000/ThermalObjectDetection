import os
import shutil

files = os.listdir("../datasets/thermal_small/images/train/")
val_files = os.listdir("../datasets/thermal_small/images/val/")

print(len(files))
print(len(val_files))
'''
os.mkdir("../datasets/thermal_small/images/val/")
os.mkdir("../datasets/thermal_small/labels/val")

for i, f in enumerate(files):
    if(not i % 10):
        shutil.move("../datasets/thermal_small/images/train/" + f, "../datasets/thermal_small/images/val/" + f)
        label_file = f.split(".")[0] + ".txt"
        shutil.move("../datasets/thermal_small/labels/train/" + label_file, "../datasets/thermal_small/labels/val/" + label_file)
'''
