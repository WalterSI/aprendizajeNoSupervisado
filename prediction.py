import numpy
import scipy.io
import random
import matplotlib.pyplot as plt

img = numpy.zeros((64, 64, 3))
  
test_data   = scipy.io.loadmat('stlTestSubset.mat')
test_images = numpy.array(test_data['testImages'])
test_labels = numpy.array(test_data['testLabels'])

opt_theta=numpy.load('opt_theta.npy')
input=numpy.load('softmax_test_data.npy')
theta = opt_theta.reshape(4, 8100)

while True:
    index=random.randint(0,3199)
    theta_x = numpy.dot(theta, input[:,index])
    hypothesis = numpy.exp(theta_x)      
    probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
    prediction = numpy.argmax(probabilities, axis = 0)
    label=test_labels[index,0]
    print "probabilidad: ", probabilities
    if (label==1):
        print "clase: avion"
    if (label==2):
	    print "clase: auto"
    if (label==3):
	    print "clase: felino"
    if (label==4):
	    print "clase: perro"
    if (prediction==1):
        print "prediccion: avion"
    if (prediction==2):
	    print "prediccion: auto"
    if (prediction==3):
	    print "prediccion: felino"
    if (prediction==4):
	    print "prediccion: perro"
    s=raw_input('--')
    img[:, :, 0] = test_images[:,:,0,index]
    img[:, :, 1] = test_images[:,:,1,index]
    img[:, :, 2] = test_images[:,:,2,index]
    plt.imshow(img)
    plt.show()
    

1=avion
2=auto
3=gato
4=perro
