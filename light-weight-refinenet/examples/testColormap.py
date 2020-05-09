from PIL import Image
import numpy as np
import cv2
import os

dir_name = "/opt/carla/PythonAPI/carla_scripts/training_data/matching_train_labels"

size = 500, 500

for line in os.listdir(dir_name):
    print(line)
    fileName = dir_name+"/"+line
    f = Image.open(fileName)
#    try:
    f.thumbnail(size, Image.ANTIALIAS)
    f.save(fileName)
#    except:
#        print("exception")
