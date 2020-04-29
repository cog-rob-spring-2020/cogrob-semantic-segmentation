import six
import sys
from models.resnet import rf_lw101, rf_lw152
from utils.helpers import prepare_img

%matplotlib inline

import glob
import os 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image

cmap = np.load('utils/cmap.npy')
has_cuda = torch.cuda.is_available()
img_dir = 'modelsSemanticSegmentation/imgs/'
imgs = glob.glob('{}*.jpg'.format(img_dir))
n_classes = 60
# Initialise models
model_inits = { 
    'RefineNet101'   : rf_lw101,
    'RefineNet152'   : rf_lw152,
    }

models = dict() 
for key,fun in six.iteritems(model_inits): # Check for cuda
    net = fun(n_classes, pretrained=True).eval()
    if has_cuda:
        net = net.cuda()
    models[key] = net

def do_calculation(images, models, gt = True):
    n_cols = len(models) + 2 # 1 - for image, 1 - for GT
    n_rows = len(images)
    if (gt == False):
        n_cols -= 1    

    plt.figure(figsize=(16, 12))
    idx = 1

    with torch.no_grad():
        for img_path in images:
            img = np.array(Image.open(img_path))
            msk = np.array(Image.open(img_path.replace('jpg', 'png')))
            orig_size = img.shape[:2][::-1]

            img_inp = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
            if has_cuda:
                img_inp = img_inp.cuda()

            plt.subplot(n_rows, n_cols, idx)
            plt.imshow(img)
            plt.title('img')
            plt.axis('off')
            idx += 1
            
            if (gt):
                plt.subplot(n_rows, n_cols, idx)
                plt.imshow(msk)
                plt.title('ground truth')
                plt.axis('off')
                idx += 1

            for mname, mnet in six.iteritems(models):
                segm = mnet(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)

                segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
                segm = cmap[segm.argmax(axis=2).astype(np.uint8)]

                plt.subplot(n_rows, n_cols, idx)
                plt.imshow(segm)
                plt.title(mname)
                plt.axis('off')
                idx += 1
                if (gt == False):
                    return segm

segm = do_calculation(imgs, models)