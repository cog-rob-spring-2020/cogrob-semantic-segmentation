import numpy as np
import os 
import cv2
from PIL import Image

dir_name = "/opt/carla/PythonAPI/carla_scripts/training_data/train_labels"
output_dir = "/opt/carla/PythonAPI/carla_scripts/training_data/matching_train_labels"

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
	image_array = cv2.resize(image_array, (int(height*factor)+1, int(width*factor)), interpolation = cv2.INTER_NEAREST)
        # print(image_array)
	for row in range(image_array.shape[0]):
		for col in range(image_array.shape[1]):
			pixel_val = image_array[row, col, chosen_index]
			if (pixel_val not in labelMap.keys()):
				labelMap[pixel_val] = counter
				counter += 1
			image_array[row, col, chosen_index] = labelMap[pixel_val]
	# image_array = cv2.resize(image_array, (0,0), fx=0.5, fy=0.5) 
#	print(image_array.shape)
	cv2.imwrite(output_dir + "/" + line, image_array[:,:,chosen_index])
	print(line)
