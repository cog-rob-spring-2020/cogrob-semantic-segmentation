import numpy as np
import os 
import cv2
from PIL import Image

dir_name = "/opt/carla/PythonAPI/carla_scripts/training_data/train_images"
output_dir = "/opt/carla/PythonAPI/carla_scripts/training_data/matching_train_images"

size = 500, 500

labelMap = {}
chosen_index = 0
counter = 0 

for line in os.listdir(dir_name):
	f =  Image.open(dir_name + "/" + line)
	#f.thumbnail(size)
	image_array = np.array(f)
	height, width = image_array.shape[0], image_array.shape[1]
	factor = 500/width
	image_array = cv2.resize(image_array[:,:,:3], (int(height*factor), int(width*factor)), interpolation = cv2.INTER_NEAREST)

	cv2.imwrite(output_dir + "/" + line, image_array)
	print("{}, {}".format(line, image_array.shape))