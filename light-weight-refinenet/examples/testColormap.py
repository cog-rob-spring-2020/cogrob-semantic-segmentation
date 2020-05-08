from PIL import Image
import numpy as np
import cv2
import os

dir_name = "/opt/carla/PythonAPI/carla_scripts/training_data/matching_train_labels"

for line in os.listdir(dir_name):
    print(line)
    fileName = dir_name+"/"+line
    f = Image.open(fileName)
    try:
        new_image = np.array(f)[:,:,1]
        cv2.imwrite(fileName, new_image)
    except:
        print("exception")
    #f.save(fileName)
