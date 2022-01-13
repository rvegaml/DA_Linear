import numpy as np
import pickle
import tensorflow as tf
from MLib.Models.KerasModels import SimpleCNNModel

class SimpleCNN():
	def __init__(self, num_units):
		self.model = SimpleCNNModel(num_units)

	def get_training_parameters(self, params):
		'''
		Function that returns the parameersneeded for the training procedure
		'''

		if 'epochs' in params:
			epochs = params['epochs']
		else:
			epochs = 500

		if 'learning_rate' in params:
			lr = params['learning_rate']
		else:
			lr = 1E-5

		if 'optimizer' in params:
			optimizer = params['optimizer']
		else:
			optimizer = tf.keras.optimizers.Adam(lr)

		if 'loss_metric' in params:
			loss_metric = params['loss_metric']
		else:
			loss_metric = tf.keras.metrics.Mean()

		return epochs, lr, optimizer, loss_metric

	def train(self, dataset, params={}, save_name='./Model_CNN.pkl'):
		'''
		Set or get the parameters of the training procedure
		'''

		epochs, lr, optimizer, loss_metric = self.get_training_parameters(params)
		loss_t_1 = np.inf
		cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

		for epoch in range(epochs):
			# print('Start of epoch %d' % (epoch,))
			for batch in dataset:
				with tf.GradientTape() as tape:
					X_batch, y_batch = batch
					z, y_hat = self.model(X_batch)

					loss = cross_entropy_loss(y_true=y_batch, y_pred=y_hat)

				grads = tape.gradient(loss, self.model.trainable_weights)

				optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

				loss_metric(loss)

			if epoch % 2 == 0:
				loss_t = loss_metric.result().numpy()
				print('Epoch ', epoch, ' Loss: ', loss_t, end='\r')

				if np.abs(loss_t - loss_t_1) < 1E-4:
					print('\n')
					print('Ending condition met. Weights converged')
					break

				loss_t_1 = loss_t

				trained_weights = self.model.get_weights()
				pickle.dump([X_batch.shape[1:3], trained_weights], open(save_name, 'wb'))

	def predict(self, X):
		return(self.model(X))

	def get_weights(self):
		return self.model.get_weights()

	def load_from_pickle(self, pickle_path):
		# Get the input size, and the trained weights
		shape, weights = pickle.load(open(pickle_path, 'rb'))

		# Make a fake prediction to initialize the model
		self.model.predict(np.zeros((1,shape[0],shape[1],1)))

		# Load the weights
		self.model.set_weights(weights)

def main():
	return -1

if __name__ == '__main__':
	main()