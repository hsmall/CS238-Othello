from Othello import *
from feature_extractor import *

import _pickle as pickle
import numpy as np
import random
from sklearn.utils import shuffle

def main():
	years = range(1977, 2018)
	#read_wtb_files(years)
	#move_counts = get_move_counts(years)
	#write_to_file(move_counts)

	'''
	states = np.concatenate( [np.load('data/states_{0}_of_5.npy'.format(i)) for i in range(1, 6)] )
	moves = np.concatenate( [np.load('data/moves_{0}_of_5.npy'.format(i)) for i in range(1, 6)] )
	
	states, moves = shuffle(states, moves)
	np.save('data/states', states, allow_pickle=False)
	np.save('data/moves', moves, allow_pickle=False)
	'''
	states = np.load('data/states.npy')
	print(states.shape)
	print('Done.')


# Reads the WTB data file for each year in |years|, converts each of them to a list
# of OthelloGames, and writes each one to the location 'data/{year}_games.pkl'
def read_wtb_files(years):
	for year in years:
		games = []
		with open('data/WTH/WTH_{0}.wtb'.format(year), 'rb') as file:
			header = file.read(16)		
			num_games = int.from_bytes(header[4:8], byteorder='little')
			print('Loading year {0} w/ {1} games'.format(year, num_games))

			for game_index in range(num_games):
				game_data = file.read(68)
				black_score = int.from_bytes(game_data[6:7], byteorder='little')

				byte_move_list = game_data[8:]
				moves = []
				for i in range(len(byte_move_list)):
					move = int.from_bytes(byte_move_list[i:i+1], byteorder='little')
					if move == 0: break
					col = move // 10 - 1
					row = move % 10 - 1
					moves.append((row, col))

				othello_game = OthelloGame(moves)
				#print(game_index, othello_game.scores, black_score)
				if black_score != othello_game.scores[OthelloGame.BLACK]:
					continue
				games.append(othello_game)

		pickle.dump(games, open('data/{0}_games.pkl'.format(year), 'wb'))


# Loads the OthelloGames for all the years in |years| and creates a map from each
# game state to a dictionary of move frequencies.
def get_move_counts(years):
	move_counts = { OthelloGame.BLACK : {}, OthelloGame.WHITE : {} }
	for year in years:
		print('Processing Year: {0}'.format(year))
		games = pickle.load(open('data/{0}_games.pkl'.format(year), 'rb'))
		for i, game in enumerate(games):
			for turn in game.turns:
				board = tuple(tuple(row) for row in turn.initial_state)
				if board not in move_counts[turn.player]:
					move_counts[turn.player][board] = { move : 1 for move in turn.legal_moves }
				move_counts[turn.player][board][turn.move] += 1
	
	return move_counts

def write_to_file(move_counts):
	num_states = len(move_counts[OthelloGame.BLACK]) +  len(move_counts[OthelloGame.WHITE])
	print('\nTotal # of States: {0}'.format(num_states))

	print('Processing/Cleaning states and moves...')
	states, moves = [], []
	count = 0
	for player in move_counts:
		for state, move_dict in move_counts[player].items():
			states.append(get_feature_matrix(state, player))
			moves.append(get_output_matrix(move_dict))

			if count % 100000 == 0:
				print('{0} out of {1}'.format(count, num_states))
			
			if count > 0 and count % 1000000 == 0:
				index = int(count / 1000000)
				print('Writing batch {0} out of 5'.format(index))
				np.save('data/states_{0}_of_5'.format(index), states, allow_pickle=False)
				np.save('data/moves_{0}_of_5'.format(index), moves, allow_pickle=False)
				del states[:]
				del moves[:]
			
			count += 1

	print('Writing batch 5 out of 5')
	np.save('data/states_5_of_5', states, allow_pickle=False)
	np.save('data/moves_5_of_5', moves, allow_pickle=False)

if __name__ == '__main__':
    main()
