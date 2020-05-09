import numpy as np
import os
from PIL import Image
names = []

lab_dir = "test_lab"
image_dir = "test_train"

train_nyu = open("test_train.nyu", "w")
val_nyu = open("test_val.nyu", "w")

for line in os.listdir("test_train"):
	f1 = Image.open("test_train/"+line)
	f2 = Image.open("test_lab/"+line)
	print("{}".format(np.array(f1).shape))
	print("vs {}".format(np.array(f2).shape))
	print(np.max(np.array(f2)))
	names.append(line)


for i in range(int(0.75*len(names))):
	image = names[i]
	print("i = {}".format(i))
	train_nyu.write(os.path.join(image_dir,image)+"\t"+os.path.join(lab_dir,image) + "\n")

for i in range(int(0.75*len(names)), len(names)):
	image = names[i]
	print("i = {}".format(i))
	val_nyu.write(os.path.join(image_dir,image)+"\t"+os.path.join(lab_dir,image) + "\n")
