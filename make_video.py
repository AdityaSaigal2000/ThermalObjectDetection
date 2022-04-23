# Author: Aditya Saigal

# Create a video from stored images.
import os
import cv2
from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

img_folder = "./test_thermal/"

# Get all files in the folder
images = os.listdir(img_folder)

# Name of output video
video_name = "thermal_upsample_faster_rcnn.avi"

# Create a frame for opencv to output to
frame = cv2.imread(os.path.join(img_folder, images[0]))

# Get the required dimension of the output images
h, w, c = frame.shape

# Construct a video writer
video = cv2.VideoWriter(video_name, 0, 1, (w, h))


for image in images:
    # Loop over images and fix their size if needed.
    # Then write them to a video. Can't adjust FPS here, need to do it later
    video.write(cv2.resize(cv2.imread(os.path.join(img_folder, image)), (w, h), interpolation = cv2.INTER_AREA))

# write initial video
cv2.destroyAllWindows()
video.release()

# Update FPS of video (speed it up) and save to a new file
in_loc = 'thermal_upsample_yolox.avi'
out_loc = 'thermal_upsample_yolox_fast.mp4'

# Import video clip
clip = VideoFileClip(in_loc)
print("fps: {}".format(clip.fps))

# Modify the FPS
clip = clip.set_fps(clip.fps * 24)

# Apply speed up
final = clip.fx(vfx.speedx, 24)
print("fps: {}".format(final.fps))

# Save video clip
final.write_videofile(out_loc)
