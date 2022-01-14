import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from keras.datasets import fashion_mnist

#Modified National Institute of Standards and Technology database
#Read the training data using loadlocal_mnist function
((X,y), (Xtest, ytest)) = keras.datasets.fashion_mnist.load_data()

#change the pixel intensities to continuous variables
X=X/255

#Display the first image
plt.imshow(X[0,:,:],cmap='gray')

#'''
##Define the model. Sequential is the easiest way to build a model in Keras.
##It allows you to build a model layer by layer
##Layer 1: a dense (fully-connected) layer with 100 hidden units. The activation
#function is ReLU.
##Layer 2: the last fully connected layer with 10 units (because we have 10 classes)
##You can use more than one fully-connected layers with different number of neurons

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

##Compile the model
##Choose the optimizer. The ADAM optimizer is selected. It is an efficient version
#of the stochastic gradient descent
##Choose the loss function. Cross-entropy is used since we have multiple calsses.
##In case we have only two classes, we can use binary_crossentropy
##When using the categorical_crossentropy loss,
#your targets should be in categorical format
#(e.g. if you have 10 classes, the target for each sample should be
#a 10-dimensional vector that is all-zeros except for a 1 at the
#index corresponding to the class of the sample).
model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''
model.compile(optimizer=tf.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
'''
#Change the pixel intensities to continuous variables
Xtest=Xtest/255

#Fit the model using X and y data. 
##One Epoch is when an ENTIRE dataset is passed forward and backward
#through the neural network only ONCE. The epoch is set to 20. 
##Since one epoch is too big to feed to the computer at once,
#we divide it to several smaller batches.
##The batch size means the number of samples in each batch. Here we set it to 100.
##verbose means how much to show the results of algorithm

model.fit(X,y,batch_size=50,epochs=10,verbose=2,validation_split=0.1)

#Compute the testing accuracy 
yhat=model.predict(Xtest)#Compute the fitted values (probabilities of classes)
#probabilites
model.evaluate(Xtest,ytest,verbose=2)


#Save your model for future use
'''
model.save('.../vanilla_model.h5')


load_model=tf.keras.models.load_model('.../vanilla_model.h5')

load_model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



im=plt.imread('...image3.jpg')
im=np.asarray(im)
im=np.mean(im,2)
im1=255-im
im1=im1/255
im1=np.reshape(im1,(1,784))
predict=load_model.predict(im1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names[np.argmax(predict)]
'''