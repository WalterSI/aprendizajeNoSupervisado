# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:59:45 2015

@author: USUARIO
"""
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

mat = scipy.io.loadmat('stlSampledPatches.mat')
#print mat

patches = np.array(mat['patches'])
print len(patches[:,0])
#print patches[:,0]

img = np.zeros((8, 8, 3))
#print img
numImage = 25
img[:, :, 0] = patches[0:64,numImage].reshape(8, 8)
img[:, :, 1] = patches[64:128,numImage].reshape(8, 8)
img[:, :, 2] = patches[128:192,numImage].reshape(8, 8)
imgplot = plt.imshow(img)
#scipy.io.savemat('prueba.mat',{'patches':img})

#prueba = scipy.io.loadmat('prueba.mat')
#prueba = np.array(prueba['patches'])
#print prueba
#imgplot = plt.imshow(prueba)
#plt.show()


