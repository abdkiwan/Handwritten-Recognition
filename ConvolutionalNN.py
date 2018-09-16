
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from mnist_loader import load_dataset

class CNN():

	def __init__(self):
		self.model = Sequential()
		 
		self.model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28,28)))
		self.model.add(Convolution2D(32, (3, 3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Dropout(0.25))
		 
		self.model.add(Flatten())
		self.model.add(Dense(128, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(10, activation='softmax'))

		self.model.compile(loss='categorical_crossentropy',
		              optimizer='adam',
		              metrics=['accuracy'])

	def train(self, train_x, train_y, epochs, batch_size):
		self.model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1)
		print("Train is complete")

	def evaluate(self, test_x, test_y):
		return self.model.evaluate(test_x, test_y, verbose=0)

	def save_model(self, model_name):
		self.model.save_weights(model_name)
		print("Saved model to disk")

	def load_model(self, model_name):
		self.model.load_weights(model_name)
		print("Loaded model from disk")

	def classify(self, x):
		return self.model.predict_classes(x)