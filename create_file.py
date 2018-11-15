import os
from random import shuffle
from eval.sintel_io import cam_read
import numpy as np

root = "/mnt/scratch/wangyang59/monkaa"
scenes = os.listdir(os.path.join(root, "frames_cleanpass"))

training_data = []
testing_data_clean = []
testing_data_final = []

all_data = []
for scene in scenes:
    frames = sorted(
        os.listdir(os.path.join(root, "frames_cleanpass", scene, "left")))
    for i in range(len(frames) - 1):
        all_data.append((scene, i))

shuffle(all_data)
training = all_data[500:]
testing = all_data[:500]

for scene in scenes:
    frames = sorted(
        os.listdir(os.path.join(root, "frames_cleanpass", scene, "left")))
    for pas in ["frames_cleanpass", "frames_finalpass"]:
        for i in range(len(frames) - 1):
            left1 = os.path.join(pas, scene, "left", frames[i])
            right1 = os.path.join(pas, scene, "right", frames[i])
            left2 = os.path.join(pas, scene, "left", frames[i + 1])
            right2 = os.path.join(pas, scene, "right", frames[i + 1])
            d = left1 + " " + right1 + " " + left2 + " " + right2 + " monkaa_cam.txt\n"
            if (scene, i) in training:
                training_data.append(d)
            elif pas == "frames_cleanpass":
                testing_data_clean.append(d)
            else:
                testing_data_final.append(d)

shuffle(training_data)

with open("monkaa_training_files.txt", 'w') as f:
    f.writelines(training_data)

with open("monkaa_testing_files_clean.txt", 'w') as f:
    f.writelines(testing_data_clean)

with open("monkaa_testing_files_final.txt", 'w') as f:
    f.writelines(testing_data_final)

## Create Sintel filenames
# root = "/mnt/scratch/wangyang59/sintel_stereo/"
# scenes = os.listdir(os.path.join(root, "clean_left"))
#   
# data = []
#   
# for scene in scenes:
#   frames = sorted(os.listdir(os.path.join(root, "clean_left", scene)))
#   for pas in ["clean_", "final_"]:
#     for i in range(len(frames)-1):
#       left1 = os.path.join(pas+"left", scene, frames[i])
#       right1 = os.path.join(pas+"right", scene, frames[i])
#       left2 = os.path.join(pas+"left", scene, frames[i+1])
#       right2 = os.path.join(pas+"right", scene, frames[i+1])
#       cam = os.path.join("depth", "camdata_left", scene, "sintel_cam.txt")
#       d = left1 + " " + right1 + " " + left2 + " " + right2 + " " + cam + "\n"
#       data.append(d)
#  
# 
# shuffle(data)
#   
# with open("sintel_files.txt", 'w') as f:
#       f.writelines(data)

## Convert sintel cam data
# with open("/mnt/data/wangyang59/kitti_zhenheng/2011_09_26/calib_cam_to_cam.txt", 'r') as f:
#   contents = f.readlines()
# 
# root = "/mnt/scratch/wangyang59/sintel_stereo/"
# scenes = os.listdir(os.path.join(root, "clean_left"))
# 
# for scene in scenes:
#   intrinsic, _ = cam_read(os.path.join(root, "depth", "camdata_left", scene, "frame_0001.cam"))
#   intrinsic = np.concatenate([intrinsic, np.zeros([3, 1])], axis=1)
#   line = "P_rect_03: " + " ".join([str(item) for item in list(intrinsic.reshape([-1]))]) + "\n"
#   new_contents = contents[0:-1] + [line]
#   with open(os.path.join(root, "depth", "camdata_left", scene, "sintel_cam.txt"), "w") as f:
#     f.writelines(new_contents)
