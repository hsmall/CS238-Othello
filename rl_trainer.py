from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from feature_extractor import *
from Othello import *
from OthelloPlayers import *
from OthelloSimulator import OthelloSimulator

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

def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def run_simulation_batch(simulator, num_simulations):
	training_set = {}

	for simulation in range(num_simulations):
		play_by_play, scores = simulator.simulate_game(cnn=True)
		winner = OthelloGame.BLACK if scores[OthelloGame.BLACK] > scores[OthelloGame.WHITE] else OthelloGame.WHITE

		for state, player, move, move_dict in play_by_play[:-1]:
			if len(move_dict) == 1: continue
			if player == winner:
				move_dict = { m : 1 if m == move else 0 for m in move_dict }
			else:
				move_dict[move] = 0
			training_set.setdefault(to_tuple(state), []).append(get_output_matrix(move_dict))
		print(scores)

	states = []
	moves = []
	for state in training_set:
		states.append(np.array(state))
		moves.append(np.mean(training_set[state], axis = 0))

	return np.array(states), np.reshape(np.array(moves), (-1,64))

def main():
	np.set_printoptions(suppress=True, precision=4)
	#states, moves = load_data()
	
	cnn = ConvolutionalNeuralNetwork(
		input_depth = 6,
		num_layers = 5,
		num_filters = 256,
		dropout_rate = 0.2,
		verbose = False,
	)
	cnn.load('models/RL_POLICY_NETWORK/RL_POLICY_NETWORK')

	simulator = OthelloSimulator(
		CNNPlayer(cnn, greedy=False),
		CNNPlayer(cnn, greedy=False)
	)

	num_simulations = 1000000
	batch_size = 10
	for simulation_batch in range(num_simulations//batch_size):
		print('Simulation Batch #{0}'.format(simulation_batch))
		states, moves = run_simulation_batch(simulator, batch_size)
		cnn.train(
			x_train = states,
			y_train = moves,
			x_valid = [],
			y_valid = [],
			batch_size=100,
			evaluation_size=0,
			num_iterations=50,
			learning_rate = 1e-6,
		)
		if (simulation_batch*batch_size) % 100 == 0:
			cnn.save('models/RL_POLICY_NETWORK/RL_POLICY_NETWORK')

if __name__ == '__main__':
    main()