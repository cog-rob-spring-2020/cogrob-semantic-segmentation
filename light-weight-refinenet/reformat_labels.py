import numpy as np
import os 
from PIL import Image
dir_name = "/opt/carla/PythonAPI/carla_scripts/training_data/matching_train_labels"


labelMap = {}
counter = 0 
for line in os.listdir(dir_name):
	f =  Image.open(dir_name + "/" + line)
	image_array = np.array(f)
	print("before", image_array)
	for row in range(image_array.shape[0]):
		for col in range(image_array.shape[1]):
			if (image_array[row, col] not in labelMap.keys()):
				labelMap[image_array[row, col]] = counter
				print("counter now {}".format(counter))
				counter += 1
			image_array[row, col] = labelMap[image_array[row, col]]
	print("after", image_array)

	print(line, "max = {}".format(max(labelMap.values())))
	break