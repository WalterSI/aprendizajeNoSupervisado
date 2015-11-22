# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:07:24 2015

@author: USUARIO
"""

# Based on CS294A/CS294W Programming Assignment Starter Code
from numpy  import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py

#def getPatches(numPatches, patchSize):

patchesPerImage = 2
numImages = 10000
numPatches = numImages*patchesPerImage
patchSize = 8
imageSize = 96


f = h5py.File('unlabeled.mat')
X=f["X"]
#print len(X[:,0])
#print X[:,0]

#images = np.zeros((imageSize, imageSize, 3))
imagesR = np.zeros((imageSize, imageSize, numImages))
imagesG = np.zeros((imageSize, imageSize, numImages))
imagesB = np.zeros((imageSize, imageSize, numImages))

for i in range(numImages):
    valor1 = imageSize*imageSize
    valor2 = valor1 + imageSize*imageSize
    valor3 = valor2 + imageSize*imageSize
    imagesR[:, :, i] = X[0:valor1,i].reshape(imageSize, imageSize)
    imagesG[:, :, i] = X[valor1:valor2,i].reshape(imageSize, imageSize)
    imagesB[:, :, i] = X[valor2:valor3,i].reshape(imageSize, imageSize)

#images[:, :, 0] = imagesR[:, :, 5]
#images[:, :, 1] = imagesG[:, :, 5]
#images[:, :, 2] = imagesB[:, :, 5]
#imgplot1 = plt.imshow(images)    

patchesR = zeros((numPatches, patchSize*patchSize))
patchesG = zeros((numPatches, patchSize*patchSize))
patchesB = zeros((numPatches, patchSize*patchSize))
	
for i in range(numImages):
    imgR = imagesR[:,:,i]
    imgG = imagesG[:,:,i]
    imgB = imagesB[:,:,i]
    
    for j in range(patchesPerImage):

        x = random.randint(imgR.shape[0] - patchSize)
        y = random.randint(imgR.shape[1] - patchSize)

        patchR = np.zeros((patchSize, patchSize, 3))
        patchG = np.zeros((patchSize, patchSize, 3))
        patchB = np.zeros((patchSize, patchSize, 3))
    
        patchR = imgR[x:x+patchSize, y:y+patchSize]
        patchG = imgG[x:x+patchSize, y:y+patchSize]
        patchB = imgB[x:x+patchSize, y:y+patchSize]
    
        patchesR[i+j, :] = patchR.reshape(1, patchSize*patchSize)
        patchesG[i+j, :] = patchG.reshape(1, patchSize*patchSize)
        patchesB[i+j, :] = patchB.reshape(1, patchSize*patchSize)

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

#print imagesPatches[:,0]
#imgplot = plt.imshow(imagesPatches)
