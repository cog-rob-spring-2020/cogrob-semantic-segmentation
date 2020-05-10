import os
import sys

sys.path += [
  "/opt/carla/PythonAPI/carla_scripts/light-weight-refinenet"
]

from RefineNet import RefineNet
import numpy as np
from PIL import Image
import time


start = time.time()

refNet = RefineNet()
init_end = time.time()

img_path = "../examples/imgs/cogProject/personInRoad.jpg"
img = np.array(Image.open(img_path))
seg = refNet.do_segmentation(np.array(Image.open(img_path))) 
segment_end = time.time() 

print(seg)
print("Init_time = {}, segment_time = {}, input.shape = {} shape = {}".format(init_end - start, segment_end-init_end, img.shape, seg.shape ))
