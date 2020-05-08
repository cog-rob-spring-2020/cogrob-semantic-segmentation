import six
import sys
from models.resnet import rf_lw101, rf_lw152
from utils.helpers import prepare_img
import glob
import os 
import cv2
import numpy as np
import torch
from PIL import Image

class RefineNet: 
	def __init__(self): 
		self.cmap = np.load('utils/cmap.npy')
		self.has_cuda = torch.cuda.is_available()
		self.n_classes = 60

		self.net = rf_lw152(self.n_classes, pretrained=True).eval()
		if self.has_cuda:
			self.net = self.net.cuda()

	def do_segmentation(self, img):
	    idx = 1

	    with torch.no_grad():
	        # img = np.array(Image.open(img_path))
	        orig_size = img.shape[:2][::-1]

	        img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
	        if self.has_cuda:
	            img_inp = img_inp.cuda()
	        

	        segm = self.net(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
	        segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
	        segm = self.cmap[segm.argmax(axis=2).astype(np.uint8)]

	        return segm
