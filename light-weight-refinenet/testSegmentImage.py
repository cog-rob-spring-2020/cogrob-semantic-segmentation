from PIL import Image 
import numpy as np

color_map = {}

f = Image.open("/opt/carla/PythonAPI/carla_scripts/training_data/train_labels/002629.png")
np_arr = np.array(f)
index_check = 0
counter = 0 
for row in range(np_arr.shape[0]):
    for col in range(np_arr.shape[1]):
        pixel_value = np_arr[row,col,index_check] 
        if (pixel_value not in color_map.keys() ):
            color_map[pixel_value] = counter
            counter += 1 


print(np_arr.shape)
print("counter = {}".format(counter))
print(color_map)
