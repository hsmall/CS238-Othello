from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Othello import *

import _pickle as pickle
import numpy as np
import random
import time

def load_data():
	print('Loading Data...')
	start_time = int(round(time.time() * 1000))
	
	states = np.load('data/states.npy')
	moves = np.reshape(np.load('data/moves.npy'), [-1, 64])
	
	end_time = int(round(time.time() * 1000))
	print('Finished in {0} seconds.\n'.format((end_time-start_time) / 1000.0))

	return states, moves

def main():
	np.set_printoptions(suppress=True, precision=4)
	states, moves = load_data()
	
	cnn = ConvolutionalNeuralNetwork(
		input_depth = states.shape[-1],
		num_layers = 5,
		num_filters = 256,
		dropout_rate = 0.05,
	)

	cnn.train(
		x_train = states[:1000000],
		y_train = moves[:1000000],
		x_valid = states[1000000:2000000],
		y_valid = moves[1000000:2000000],
		batch_size=1000,
		evaluation_size=10000,
		num_iterations=30000,
		model_name = "models/SL_POLICY_NETWORK"
	)

if __name__ == '__main__':
    main()