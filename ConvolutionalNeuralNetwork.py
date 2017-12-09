from Othello import *

import numpy as np
import os
import random
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress Logging/Warnings


class ConvolutionalNeuralNetwork:
	def __init__(self, input_depth=None, num_layers=2, num_filters=256, dropout_rate=0.5, verbose=True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.board_size = OthelloGame.BOARD_SIZE
			self.input_depth = input_depth
			self.is_training = tf.placeholder(tf.bool)
			
			self.x = tf.placeholder(tf.float32, [None, self.board_size, self.board_size, self.input_depth], name="x")
			self.y = tf.placeholder(tf.float32, [None, self.board_size*self.board_size], name="y")
			self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
			if verbose: print("Input Shape: {0}".format(self.x.shape))

			layers = [self.x]
			for i in range(num_layers):
				layers.append(tf.layers.dropout(
					inputs = tf.layers.conv2d(
						inputs = layers[-1],
						filters = num_filters,
						kernel_size = 5,
						padding = "same",
						kernel_initializer = tf.contrib.layers.xavier_initializer(),
						bias_initializer = tf.contrib.layers.xavier_initializer(),
						activation = tf.nn.relu,
						name = "conv{0}".format(i+1)
					),
					rate = dropout_rate,
					training = self.is_training,
					name = "dropout{0}".format(i+1)
				))
				if verbose: print("Hidden Layer #{0} Output Shape: {1}".format(i+1, layers[-1].shape))
			
			self.final_conv = tf.layers.conv2d(
				inputs = layers[-1],
				filters = 1,
				kernel_size = 1,
				padding = "same",
				kernel_initializer = tf.contrib.layers.xavier_initializer(),
				bias_initializer = tf.contrib.layers.xavier_initializer(),
				activation = tf.nn.relu,
				name = "final_conv"
			)
			if verbose: print("Final Layer Output Shape: {0}\n".format(self.final_conv.shape))

			self.output = tf.reshape(self.final_conv, [-1, self.board_size*self.board_size])
			self.softmax_output = tf.nn.softmax(self.output)

			self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.output))
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output, axis = 1), tf.argmax(self.y, axis = 1)), tf.float32) )
			self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.error)

			self.session = tf.Session()
			self.session.run(tf.global_variables_initializer())

	def save(self, filename):
		with self.graph.as_default():
			self.saver = tf.train.Saver()
			self.saver.save(self.session, filename)

	def load(self, filename):
		with self.graph.as_default():
			self.saver = tf.train.Saver()
			self.saver.restore(self.session, filename)

	def predict(self, data, softmax=True):
		data_reshaped = np.reshape(data, [-1, self.board_size, self.board_size, self.input_depth])
		output = self.session.run(self.softmax_output, feed_dict = {self.x: data_reshaped, self.is_training: False})
		return np.reshape(output, [-1, self.board_size, self.board_size])

	def train(self, x_train, y_train, x_valid, y_valid,
				    batch_size = 20,
				    evaluation_size = 100,
				    num_iterations = 30000,
				    learning_rate = 1e-4,
				    model_name = None):

		print ("Training Set Size: {0}, Validation Set Size: {1}".format(len(y_train), len(y_valid)))
		lowest_error = float("inf")
		lowest_error_iteration = 0
		for iteration in range(num_iterations + 1):
			batch_indices = random.sample(range(len(x_train)), batch_size)
			x_batch = x_train[batch_indices]
			y_batch = y_train[batch_indices]

			if evaluation_size > 0 and iteration % 50 == 0:
				evaluation_indices = random.sample(range(len(x_train)), evaluation_size)
				train_error, train_accuracy = self.session.run((self.error, self.accuracy),
															   feed_dict = {self.x: x_train[evaluation_indices],
															            	self.y: y_train[evaluation_indices],
															            	self.is_training: False})
				
				evaluation_indices = random.sample(range(len(x_valid)), evaluation_size)
				valid_error, valid_accuracy = self.session.run((self.error, self.accuracy),
															   feed_dict = {self.x: x_valid[evaluation_indices],
															            	self.y: y_valid[evaluation_indices],
															            	self.is_training: False})
				marker = ""
				if model_name is not None and valid_error <= lowest_error:
					lowest_error = valid_error
					lowest_error_iteration = iteration
					self.save(model_name)
					marker = "*"
				
				print("Iteration: {0:>4} -> Train: {1:.3f} ({2:.3f}), Validation: {3:.3f} ({4:.3f}) {5}".format(iteration,
																											train_error,
																											train_accuracy,
																											valid_error,
																											valid_accuracy,
																											marker))
				
			self.session.run(self.train_step, feed_dict = {self.x: x_batch,
														   self.y: y_batch,
														   self.learning_rate: learning_rate,
														   self.is_training: True})



