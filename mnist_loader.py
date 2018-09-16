import cPickle, gzip, numpy

from keras.utils import np_utils
from keras.utils import np_utils

def load_dataset():
	# Load the dataset
	f = gzip.open('mnist.pkl.gz', 'rb')
	train_data, _, test_data = cPickle.load(f)
	f.close()

	train_x, train_y = train_data
	test_x, test_y = test_data

	train_x = train_x.reshape(train_x.shape[0], 1, 28, 28)
	test_x = test_x.reshape(test_x.shape[0], 1, 28, 28)

	# Convert 1-dimensional class arrays to 10-dimensional class matrices
	train_y = np_utils.to_categorical(train_y, 10)
	test_y = np_utils.to_categorical(test_y, 10)

	return train_x, train_y, test_x, test_y