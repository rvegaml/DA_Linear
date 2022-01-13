import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import glorot_normal

class LinearLayer(layers.Layer):
	'''
	A simple linear layer that penalizes deviations from the initial weights
	'''

	# According to the TensorFlow documentation, it's a good practice to add this function
	def __init__(self, units=10, **kwargs):
		super(LinearLayer, self).__init__(**kwargs)
		self.units = units

	# According to the TensorFlow documentation, it's a good practice to add this function
	def get_config(self):
		config = super(LinearLayer, self).get_config()
		return config


	def build(self, input_shape):
		# Get the number of dimensions of the data
		num_dim = input_shape[-1]

		# Build the actual weights
		self.w = self.add_weight(shape=(num_dim, self.units),
			initializer='random_normal',
			trainable=True)
		self.b = self.add_weight(shape=(self.units,),
			initializer='random_normal',
			trainable=True)

	def call(self, inputs):

		return tf.matmul(inputs, self.w) + self.b
		
class ConvLayer(layers.Layer):
	'''
	Layer that computes the 2D convolution and penalizes deviations from weights.
	'''

	def __init__(self, size=[3,3], num_filters=32, gate=tf.nn.relu, 
		stride=[1,1,1,1], padding='SAME', **kwargs):
	
		super(ConvLayer, self).__init__(**kwargs)

		self.size = size
		self.num_filters = num_filters
		self.gate = gate
		self.stride = stride
		self.padding = padding

	# According to the TensorFlow documentation, it's a good practice to add this function
	def get_config(self):
		config = super(ConvLayer, self).get_config()
		return config

	def build(self, input_shape):
		# Get the number of dimensions of the data
		dim_in = input_shape[-1]
		filter_height = self.size[0]
		filter_width = self.size[1]

		# Build the actual weights
		self.w = self.add_weight(shape=(filter_height, filter_width, dim_in, self.num_filters),
			initializer=glorot_normal(),
			trainable=True)
		self.b = self.add_weight(shape=(self.num_filters,),
			initializer=glorot_normal(),
			trainable=True)

	def call(self, inputs):
		
		x = tf.nn.conv2d(inputs, filters=self.w, strides=self.stride, padding=self.padding)
		x = tf.add(x, self.b)

		return self.gate(x)

def main():
	return -1

if __name__ == '__main__':
	main()