# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:07:24 2015

@author: USUARIO
"""

# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io
import h5py
import datetime

#def getPatches(numPatches, patchSize):

dateInit = datetime.datetime.now()
patchesPerImage = 2
numImages = 100000
numPatches = numImages*patchesPerImage
patchSize = 8
imageSize = 96


f = h5py.File('unlabeled.mat')
X=f["X"]

patchesR = zeros((numPatches, patchSize*patchSize))
patchesG = zeros((numPatches, patchSize*patchSize))
patchesB = zeros((numPatches, patchSize*patchSize))
	
for i in range(numImages):
    
    valor1 = imageSize*imageSize
    valor2 = valor1 + imageSize*imageSize
    valor3 = valor2 + imageSize*imageSize
    imgR = X[0:valor1,i].reshape(imageSize, imageSize)
    imgG = X[valor1:valor2,i].reshape(imageSize, imageSize)
    imgB = X[valor2:valor3,i].reshape(imageSize, imageSize)
    
    for j in range(patchesPerImage):

        x = random.randint(imgR.shape[0] - patchSize)
        y = random.randint(imgR.shape[1] - patchSize)

        patchesR[i+j, :] = imgR[x:x+patchSize, y:y+patchSize].reshape(1, patchSize*patchSize)
        patchesG[i+j, :] = imgG[x:x+patchSize, y:y+patchSize].reshape(1, patchSize*patchSize)
        patchesB[i+j, :] = imgB[x:x+patchSize, y:y+patchSize].reshape(1, patchSize*patchSize)

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
#imgplot2 = plt.imshow(patch)

imagesPatches = np.zeros((patchSize*patchSize*3, numPatches))
for i in range(numPatches):
    imagesPatches[0:64, i] = patchesR[i,:]
    imagesPatches[64:128, i] = patchesG[i,:]
    imagesPatches[128:192, i] = patchesB[i,:]

scipy.io.savemat('walterPatches.mat',{'patches':imagesPatches})
duration = datetime.datetime.now() - dateInit
print duration
#print imagesPatches[:,0]
#imgplot = plt.imshow(imagesPatches)
