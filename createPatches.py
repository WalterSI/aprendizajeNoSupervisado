# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:07:24 2015
@author: Walter Sotomayor
"""

from numpy  import *
import numpy as np
import scipy.io
import datetime


dateInit = datetime.datetime.now()
patchesPerImage = 1
numImages = 100000
numPatches = numImages*patchesPerImage
patchSize = 8
imageSize = 96

patchesR = zeros((numPatches, patchSize*patchSize))
patchesG = zeros((numPatches, patchSize*patchSize))
patchesB = zeros((numPatches, patchSize*patchSize))
	
train_X_file =  open("./data/stl10_binary/unlabeled_X.bin", "rb")
data = np.fromfile(train_X_file, dtype=np.uint8)
data = data.reshape(100000, 3, 96*96)

for i in range(numImages):
    
    img = np.column_stack([data[i][0], data[i][1], data[i][2]])
    img = img.reshape(96, 96, 3)
    
    for j in range(patchesPerImage):
        x = random.randint(img.shape[0] - patchSize)
        y = random.randint(img.shape[1] - patchSize)
        imgR = img[x:x+patchSize, y:y+patchSize, 0]
        imgG = img[x:x+patchSize, y:y+patchSize, 1]
        imgB = img[x:x+patchSize, y:y+patchSize, 2]
        patchesR[i+j, :] = imgR.reshape(1, patchSize*patchSize)
        patchesG[i+j, :] = imgG.reshape(1, patchSize*patchSize)
        patchesB[i+j, :] = imgB.reshape(1, patchSize*patchSize)

print "paso for"
# Remove DC (mean of images)
patchesR = patchesR - mean(patchesR)
patchesG = patchesG - mean(patchesG)
patchesB = patchesB - mean(patchesB)
	
# Truncate to +/-3 standard deviations and scale to -1 to 1
pstdR = 3 * std(patchesR)
patchesR = maximum(minimum(patchesR, pstdR), -pstdR) / pstdR
pstdG = 3 * std(patchesG)
patchesG = maximum(minimum(patchesG, pstdG), -pstdG) / pstdG
pstdB = 3 * std(patchesB)
patchesB = maximum(minimum(patchesB, pstdB), -pstdB) / pstdB
	
# Rescale from [-1,1] to [0.1,0.9]
patchesR = (patchesR + 1) * 0.4 + 0.1
patchesG = (patchesG + 1) * 0.4 + 0.1
patchesB = (patchesB + 1) * 0.4 + 0.1

imagesPatches = np.zeros((patchSize*patchSize*3, numPatches))
for i in range(numPatches):
    imagesPatches[0:64, i] = patchesR[i,:]
    imagesPatches[64:128, i] = patchesG[i,:]
    imagesPatches[128:192, i] = patchesB[i,:]

scipy.io.savemat('walterPatches.mat',{'patches':imagesPatches})
duration = datetime.datetime.now() - dateInit
print duration
