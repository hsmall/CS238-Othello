from Othello import *
import numpy as np

POSITION_WEIGHTS = [
	[20, -3, 11,  8,  8, 11, -3, 20],
	[-3, -7, -4,  1,  1, -4, -7, -3],
	[11, -4,  2,  2,  2,  2, -4, 11],
	[ 8,  1,  2, -3, -3,  2,  1,  8],
	[ 8,  1,  2, -3, -3,  2,  1,  8],
	[11, -4,  2,  2,  2,  2, -4, 11],
	[-3, -7, -4,  1,  1, -4, -7, -3],
	[20, -3, 11,  8,  8, 11, -3, 20]
]

def get_feature_matrix(input_state, player):
	state = np.array(input_state)
	turn = OthelloTurn(state, player, (0,0))
	
	captured_pieces = np.zeros((OthelloGame.BOARD_SIZE, OthelloGame.BOARD_SIZE))
	weighted_captured_pieces = np.zeros((OthelloGame.BOARD_SIZE, OthelloGame.BOARD_SIZE))
	next_state_scores = np.zeros((OthelloGame.BOARD_SIZE, OthelloGame.BOARD_SIZE))
	weighted_next_state_scores = np.zeros((OthelloGame.BOARD_SIZE, OthelloGame.BOARD_SIZE))
	
	for ((row, col), pieces) in turn.legal_moves.items():
		captured_pieces[row, col] = len(pieces) + 1

		weighted_score = sum(POSITION_WEIGHTS[r][c] for r, c in pieces)
		weighted_captured_pieces[row, col] = weighted_score + POSITION_WEIGHTS[row][col]

		next_state = np.array(turn.make_move(state, player, (row, col))) * player
		next_state_scores[row, col] = np.sum(next_state)
		weighted_next_state_scores[row, col] = np.sum(np.multiply(next_state, POSITION_WEIGHTS))

	return np.dstack((
		state * player, 			# +1 for current player pieces, -1 for opponent pieces.
		captured_pieces, 			# Number of pieces that are captured by each legal move.
		weighted_captured_pieces, 	# captured_pieces weighted by POSITION_WEIGHTS.
		next_state_scores, 			# Difference between number of player pieces and number of opponent pieces.
		weighted_next_state_scores,	# next_state_scores weighted by POSITION_WEIGHTS.
		POSITION_WEIGHTS			# The value that each position has on the Othello board.
	))

def get_output_matrix(move_dict):
	grid = np.zeros((OthelloGame.BOARD_SIZE, OthelloGame.BOARD_SIZE))
	for ((row, col), count) in move_dict.items():
		grid[row,col] = count

	grid /= grid.sum()
	return grid 