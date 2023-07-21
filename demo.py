#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import model
import time


#load model
testmodel = model.JWSTFixModel()
testmodel.load_state_dict(torch.load('model_484.pth'))
testmodel.eval()

#load / resize img
Image.MAX_IMAGE_PIXELS, res = 99999999999, 2048
pth = 'examples/in.jpg'
img = Image.open(pth)
img.thumbnail((res, res))

#infer
with torch.no_grad():
    pred = testmodel(T.ToTensor()(img).unsqueeze(0))

#save img
pred = Image.fromarray((model.pt_to_np(pred[0])*255).astype(np.uint8))
plt.imshow(pred)
plt.show()
pred.save(f'out_{int(time.time()*100)}.jpg')























