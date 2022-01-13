import numpy as np
import tensorflow as tf
import pickle
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from MLib.Core.losses import MMD, MMD_tf
from MLib.Core.kernels import gaussian_kernel

def exp_norm(X_broad, Y_broad, sigma):
	'''
	X_broad is a matrix of NumElemX x Dim x NumElemY
	Y_boad is a matrix of NumElemX x Dim x NumElemY
	sigma is a scalar
	'''
	N, d, M = X_broad.get_shape().as_list()
	
	# Get the subtraction of every possible pair between X and Y
	subtraction = tf.math.subtract(X_broad, Y_broad)
	
	# Make the matrix N X M X D
	subtraction = tf.transpose(subtraction, (0, 2, 1))

	# Get the squared norm of the difference between each pair of vectors n X and Y
	norm_squared = tf.math.square(tf.norm(subtraction, axis=-1))
	
	# Compute the exponential operation
	denominator = 2 * sigma * sigma
	exp_arg = tf.negative(tf.divide(norm_squared, denominator))
	exp = tf.math.exp(exp_arg)
	
	# Transform to a matrix of N*M x 1
	reshaped_norm = tf.reshape(exp, (N*M, 1))
	
	return reshaped_norm

def compute_outer_Y_XT(X_broad, Y_broad):
	'''
	computes all the outer products between the vectors in X_broad and Y_broad
	X_broad and Y_broad are matrices of size NumElemX, d, NumElemY
	'''
	N, d, M = X_broad.get_shape().as_list()
	
	new_X = tf.transpose(X_broad, (1,0,2)) # d x N x M
	new_Y = tf.transpose(Y_broad, (1,0,2)) # d x N x M

	# Order all the vectors x as row vectors
	x_vectors_row = tf.reshape(new_X, [1, d, N*M])
	
	# Order all the vectors y as column vectos:
	y_vectors_col = tf.reshape(new_Y, [d, 1, N*M])
	
	# Compute the outer products in parallel
	x_vectors_row = tf.transpose(x_vectors_row, (2,0,1))
	y_vectors_col = tf.transpose(y_vectors_col, (2,0,1))

	outer_y_xT = tf.linalg.matmul(y_vectors_col, x_vectors_row)
	
	return outer_y_xT

def grad_MMD_tf_simple(X, Y, sigma, R):
	'''
	Evaluates the gradient of the MMD w.r.t. the rotation matrix R
	'''
	# Get the number of instances in each batch
	N, d = X.get_shape().as_list()
	M, d = Y.get_shape().as_list()
	
	# Apply the Rotation to Y
	rot_Y = tf.linalg.matmul(Y, tf.transpose(R))
	
	# ------------------------------------------
	# Compute the gradient of the first term wrt R
	# ------------------------------------------
	
	# This gradient is always 0
	
	#-------------------------------------------
	# Compute the gradient of the second term
	# ------------------------------------------
	
	# Make copies of the vectors to make the computation more efficient.
	X_broad = tf.broadcast_to(tf.expand_dims(X,2), [N, d, M])
	
	Y_broad = tf.broadcast_to(tf.expand_dims(Y,2), [M, d, N])
	Y_trans = tf.transpose(Y_broad, (2,1,0))
	
	rot_Y_broad = tf.broadcast_to(tf.expand_dims(rot_Y,2), [M, d, N])
	rot_Y_trans = tf.transpose(rot_Y_broad, (2,1,0))
	
	# Apply the Gaussian Kernel to every pair of vectors
	reshaped_exp_norm = exp_norm(X_broad, rot_Y_trans, sigma)
	
	# Broadcast to make further computations more efficient
	exp_norm_broad = tf.broadcast_to(tf.expand_dims(reshaped_exp_norm,2), 
								[M*N, d, d] )
		
	# Compute all the outer products < y_j, x_i^T >
	outer_y_xT = tf.negative(compute_outer_Y_XT(X_broad, Y_trans))
	outer_x_yT = tf.transpose(outer_y_xT, [0, 2, 1])
	
	
	# Divide by -sigma^2
	denominator = sigma * sigma
	normalized_sum = tf.negative(tf.divide(outer_x_yT, denominator))
	
	# Element-wise multiplication of the normalized sum and the exponential term
	element_mult = tf.math.multiply(exp_norm_broad, normalized_sum)
	
	# Sum across the main axis
	second_term_sum = tf.math.reduce_sum(element_mult, axis=0)
	
	# Compute the normalization constant
	c = tf.divide(-2, tf.math.multiply(M, N))
	
	# Compute the final gradient of the second term
	g_second_term = tf.multiply(c, second_term_sum)
	
	
	#-------------------------------------------
	# Compute the gradient of the third term
	# ------------------------------------------
	# This gradient is always 0
	
	return g_second_term

def compute_A(G, R):
	'''
	G is the gradient of the cost function to optimize
	R is a feasible point
	'''
	term_1 = tf.linalg.matmul(G, tf.transpose(R))
	term_2 = tf.linalg.matmul(R, tf.transpose(G))
	
	A = tf.math.subtract(term_1, term_2)
	
	return A


def compute_update_simple(X, Y, sigma, R, tau):
	# Compute the gradient
	G = grad_MMD_tf_simple(X, Y, sigma, R)
	
	# Estimate the matrix A
	A = compute_A(G, R)
	
	# Create the identity matrix
	I = tf.eye(A.shape[0], dtype=tf.float64)

	# Compute the update
	term_1 = tf.math.add(I, tf.math.multiply(0.5*tau, A))
	inv_term_1 = tf.linalg.inv(term_1)
	term_2 = tf.math.subtract(I, tf.math.multiply(0.5*tau, A))
	
	Q = tf.linalg.matmul(inv_term_1, term_2)
	
	new_R = tf.linalg.matmul(Q, R)
	
	return new_R, G.numpy()

	
def find_optimal_rotation_simple(X, Y, sigma, R_init, tau, max_iter=100, max_iter_learning_rate=100):
	# Convert the numpy arrays to tensors
	X_tensor = tf.convert_to_tensor(X)
	Y_tensor = tf.convert_to_tensor(Y)
	c_R = tf.convert_to_tensor(R_init)
	c_G = 0.0
	# Start by computing the current MMD between the two samples
	c_MMD = MMD_tf(X_tensor, tf.linalg.matmul(Y_tensor, tf.transpose(c_R)), sigma)
	c_MMD_1 = MMD_tf(X_tensor, tf.linalg.matmul(Y_tensor, tf.transpose(c_R)), sigma)
	print('Initial MMD')
	print(c_MMD)
	print('\n')
	
	# Apply the updates and check if the MMD goes down
	for i in range(max_iter):
		found_tau = False
		
		for j in range(max_iter_learning_rate):
			possible_R, possible_G = compute_update_simple(X_tensor, Y_tensor, sigma, c_R, tau)
			possible_MMD = MMD_tf(
				X_tensor, 
				tf.linalg.matmul(Y_tensor, tf.transpose(possible_R)), 
				sigma
			)
			
			if possible_MMD.numpy() < c_MMD.numpy():
				c_R = possible_R
				c_MMD = possible_MMD
				c_G = possible_G
				found_tau = True
				print(c_MMD.numpy(), end='\r')
				break
			else:
				tau = 0.5*tau
		
		if np.abs(c_MMD.numpy() - c_MMD_1.numpy()) < 1E-6:
			print('Small change in MMD. Stopping.')
			break
		else:
			c_MMD_1 = c_MMD
		if found_tau == False:
			break
	
	return c_R, c_MMD

def find_optimal_rotation_batches(X, Y, sigma, R_init, tau, max_iter=100, max_iter_learning_rate=100, batch_size=100):
	num_X = X.shape[0]
	num_Y = Y.shape[0]	

	random_order_X = np.random.permutation(num_X)[0:batch_size]
	random_order_Y = np.random.permutation(num_Y)[0:batch_size]

	# Convert the numpy arrays to tensors
	X_tensor = tf.convert_to_tensor(X[random_order_X])
	Y_tensor = tf.convert_to_tensor(Y[random_order_Y])
	c_R = tf.convert_to_tensor(R_init)
	c_G = 0.0
	# Start by computing the current MMD between the two samples
	c_MMD = MMD_tf(X_tensor, tf.linalg.matmul(Y_tensor, tf.transpose(c_R)), sigma)
	c_MMD_1 = MMD_tf(X_tensor, tf.linalg.matmul(Y_tensor, tf.transpose(c_R)), sigma)
	print('Initial MMD')
	print(c_MMD)
	print('\n')
	
	# Apply the updates and check if the MMD goes down
	for i in range(max_iter):
		found_tau = False
		
		random_order_X = np.random.permutation(num_X)
		random_order_Y = np.random.permutation(num_Y)

		# Convert the numpy arrays to tensors
		X_tensor = tf.convert_to_tensor(X[random_order_X])[0:batch_size]
		Y_tensor = tf.convert_to_tensor(Y[random_order_Y])[0:batch_size]

		# Start by computing the current MMD between the two samples
		c_MMD = MMD_tf(X_tensor, tf.linalg.matmul(Y_tensor, tf.transpose(c_R)), sigma)
		c_MMD_1 = MMD_tf(X_tensor, tf.linalg.matmul(Y_tensor, tf.transpose(c_R)), sigma)

		for j in range(max_iter_learning_rate):
			possible_R, possible_G = compute_update_simple(X_tensor, Y_tensor, sigma, c_R, tau)
			possible_MMD = MMD_tf(
				X_tensor, 
				tf.linalg.matmul(Y_tensor, tf.transpose(possible_R)), 
				sigma
			)
			
			if possible_MMD.numpy() < c_MMD.numpy():
				c_R = possible_R
				c_MMD = possible_MMD
				c_G = possible_G
				found_tau = True
				print(c_MMD.numpy(), end='\r')
				break
			else:
				tau = 0.5*tau
		
		if np.abs(c_MMD.numpy() - c_MMD_1.numpy()) < 1E-6:
			print('Small change in MMD. Stopping.')
			break
		else:
			c_MMD_1 = c_MMD
		if found_tau == False:
			break
	
	return c_R, c_MMD

def main():
	return -1

if __name__ == '__main__':
	main()