import numpy as np
import tensorflow as tf
from MLib.Core.layers import LinearLayer, ConvLayer
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras import Model

class SimpleCNNModel(Model):
	def __init__(self, num_units):

		super(SimpleCNNModel, self).__init__()

		# Define the architecture of the network
		self.conv1 = ConvLayer(size=[3,3], num_filters=32, gate=tf.nn.relu)
		self.pool1 = MaxPooling2D(pool_size=[2,2])

		self.conv2 = ConvLayer(size=[3,3], num_filters=128, gate=tf.nn.relu)
		self.pool2 = MaxPooling2D(pool_size=[2,2])

		self.conv3 = ConvLayer(size=[3,3], num_filters=256, gate=tf.nn.relu)
		self.pool3 = MaxPooling2D(pool_size=[2,2])

		self.conv4 = ConvLayer(size=[3,3], num_filters=512, gate=tf.nn.relu)
		self.pool4 = MaxPooling2D(pool_size=[2,2])

		self.flat = Flatten()

		self.hidden = LinearLayer(units=num_units)

		self.final = LinearLayer(units=10)

	def call(self, inputs):

		# First conv-pooling layer
		x = self.conv1(inputs)
		x = self.pool1(x)

		# Second conv-pooling layer
		x = self.conv2(x)
		x = self.pool2(x)

		# Third conv-pooling layer
		x = self.conv3(x)
		x = self.pool3(x)

		# Fourth conv-pooling layer
		x = self.conv4(x)
		x = self.pool4(x)

		# Flatten the array 
		x = self.flat(x)

		# First fully connected layer
		z = self.hidden(x)
		z = tf.nn.relu(z)

		# Prediction layer
		x = self.final(z)
		x = tf.nn.softmax(x)

		return z, x

def main():
	return -1

if __name__ == '__main__':
	main()