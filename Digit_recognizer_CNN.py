import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
	# create model
	model = Sequential() 
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))  #The layer has 32 feature maps, which with the size of 5×5 and a rectifier activation function.
	model.add(MaxPooling2D(pool_size=(2, 2)))    #pooling layer that takes the max called MaxPooling2D. It is configured with a pool size of 2×2.
	model.add(Dropout(0.2))  #This dropout layer is configured to randomly exclude 20% of neurons in the layer in order to reduce overfitting.
	
	'''If you wanted to use a Dense(a fully connected layer) after your convolution layers, 
	you would need to ‘unstack’ all this multidimensional tensor into a very long 1D tensor.
	 You can achieve this using Flatten.'''

	model.add(Flatten())	# It is a layer that converts the 2D matrix data to a vector called Flatten. It allows the output to be processed by standard fully connected layers.
	model.add(Dense(128, activation='relu'))  #Next a fully connected layer with 128 neurons and rectifier activation function.
	model.add(Dense(num_classes, activation='softmax')) #Finally, the output layer has 10 neurons for the 10 classes and a softmax activation function to output probability-like predictions for each class.

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))







