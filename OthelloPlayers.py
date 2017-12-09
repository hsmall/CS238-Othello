from Othello import *


import random
class RandomPlayer:
	def __init__(self):
		pass
	
	def set_color(self, color):
		self.color = color

	def select_move(self, board, player, legal_moves):
		return random.choice(legal_moves), None, None


from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from feature_extractor import *
class CNNPlayer:
	def __init__(self, cnn, greedy=False):
		self.cnn = cnn
		self.greedy = greedy

	def set_color(self, color):
		self.color = color

	def normalize(self, array):
		total = sum(array)
		return [elem / total for elem in array]

	def select_move(self, board, player, legal_moves):
		input_state = get_feature_matrix(board, player)
		prob_distr = self.cnn.predict(input_state)[0,:,:]
		move_probs = self.normalize([ prob_distr[row, col] for row, col in legal_moves ])
		
		if not self.greedy:
			move_index = np.random.choice(range(len(legal_moves)), 1, p=move_probs)[0]
		else:
			move_index = np.argmax(move_probs)
		
		move_dict = { legal_moves[i] : move_probs[i] for i in range(len(legal_moves)) }
		return legal_moves[move_index], input_state, move_dict


class MiniMaxPlayer:
	def __init__(self, value_fn, depth=2):
		self.value_fn = value_fn
		self.depth = depth

	def set_color(self, color):
		self.color = color

	def select_move(self, board, player, legal_moves):
		value, move = self.minimax(board, player, self.depth)
		return move, None, None

	def get_legal_moves(self, board, player):
		return list(OthelloTurn.get_legal_moves(board, player).keys())

	def minimax(self, board, player, depth):
		legal_moves = self.get_legal_moves(board, player)
		#print(legal_moves)
		if depth == 0 or len(legal_moves) == 0: return (self.value_fn(board, self.color), None)

		next_depth = depth-1 if player != self.color else depth
		move_choices = [ (self.minimax(OthelloTurn.make_move(board, player, move), -player, next_depth)[0], move) for move in legal_moves ]

		if player == self.color:
			optimal_value = max(move_choices)[0]
		else:
			optimal_value = min(move_choices)[0]

		optimal_moves = [(value, move) for value, move in move_choices if value == optimal_value]
		result = random.choice(optimal_moves)
		#print(result)
		return result
