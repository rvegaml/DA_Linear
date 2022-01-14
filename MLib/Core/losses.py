'''
File: losses.py
Description:
	This file contains common loss functions used in my functions.
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def sum_gaussian_kernels(X, Y, sigma):
	'''
	This function gets two tensors of shape num_instances x num_dimenstions and get
	the sum between every pair of elements in X and Y, after applying the gaussian kernel
	using the provided sigma.
	'''

	# Get the number of instances in each batch
	dims_X = X.get_shape().as_list()
	dims_Y = Y.get_shape().as_list()

	# Get the subtraction of every possible pair between X and Y
	X_broad = tf.broadcast_to(tf.expand_dims(X,2), [dims_X[0], dims_X[1], dims_Y[0]])
	Y_broad = tf.broadcast_to(tf.expand_dims(Y,2), [dims_Y[0], dims_Y[1], dims_X[0]])
	Y_trans = tf.transpose(Y_broad, (2,1,0))

	subtraction = tf.math.subtract(X_broad, Y_trans)

	# Make the matrix N X M X D
	subtraction = tf.transpose(subtraction, (0, 2, 1))

	# Get the squared norm of the difference between each pair of vectors n X and Y
	norm_squared = tf.math.square(tf.norm(subtraction, axis=-1))

	# Compute the exponential operation
	denominator = 2 * sigma * sigma
	exp_arg = tf.negative(tf.divide(norm_squared, denominator))
	exp = tf.math.exp(exp_arg)

	# Compute the sum of all possible pairs
	sum_gaussian_kernel = tf.math.reduce_sum(exp)

	normalized_sum = tf.divide(sum_gaussian_kernel, dims_X[0]*dims_Y[0])

	return normalized_sum


def MMD_tf(X, Y, sigma):
	first_term = sum_gaussian_kernels(X,X, sigma)
	second_term = sum_gaussian_kernels(X,Y, sigma)
	third_term = sum_gaussian_kernels(Y,Y, sigma)

	return first_term - 2*second_term + third_term

def main():
	return -1

if __name__ == '__main__':
	main()
