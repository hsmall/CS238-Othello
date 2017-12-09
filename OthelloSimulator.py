from copy import deepcopy

from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Othello import *

from OthelloPlayers import *
from feature_extractor import *


class OthelloSimulator:
	def __init__(self, player1, player2):
		self.players = {OthelloGame.BLACK : player1, OthelloGame.WHITE : player2}
		player1.set_color(OthelloGame.BLACK)
		player2.set_color(OthelloGame.WHITE)

	def simulate_game(self, cnn=False):
		play_by_play = []
		
		board = OthelloGame.INITIAL_BOARD
		player = OthelloGame.BLACK

		while True:
			legal_moves = self.get_legal_moves(board, player)
			if len(legal_moves) == 0:
				player *= -1
				legal_moves = self.get_legal_moves(board, player)
				if len(legal_moves) == 0:
					break

			move, state, move_dict = self.players[player].select_move(board, player, legal_moves)
			if cnn:
				play_by_play.append((state, player, move, move_dict))
			else:
				play_by_play.append((board, player, move))
			
			board = OthelloTurn.make_move(board, player, move)
			player *= -1

		play_by_play.append((board, None, None))
		return play_by_play, OthelloGame.compute_scores(board)

	def get_legal_moves(self, board, player):
		return list(OthelloTurn.get_legal_moves(board, player).keys())


def main():
	np.set_printoptions(suppress=True, precision=4)
	
	sl_policy_network = ConvolutionalNeuralNetwork(
		input_depth = 6,
		num_layers = 5,
		num_filters = 256,
		dropout_rate = 0.0,
		verbose = False,
	)
	sl_policy_network.load("models/SL_POLICY_NETWORK/SL_POLICY_NETWORK")

	rl_policy_network = ConvolutionalNeuralNetwork(
		input_depth = 6,
		num_layers = 5,
		num_filters = 256,
		dropout_rate = 0.0,
		verbose = False,
	)
	rl_policy_network.load("models/RL_POLICY_NETWORK/RL_POLICY_NETWORK")

	random_player = RandomPlayer()
	heuristic_player = MiniMaxPlayer(lambda board, player: np.sum(board)*player, depth=2)
	sl_player = CNNPlayer(sl_policy_network, greedy=True)
	rl_player = CNNPlayer(rl_policy_network, greedy=True)

	simulator = OthelloSimulator(random_player, heuristic_player)
	win_count = 0.0
	num_simulations = 5
	all_scores = []
	for i in range(num_simulations):
		play_by_play, scores = simulator.simulate_game()
		if scores[OthelloGame.BLACK] > scores[OthelloGame.WHITE]:
			win_count += 1
		print(scores[OthelloGame.BLACK])
		all_scores.append(scores[OthelloGame.BLACK])

	print("Win Percentage: {0:.2f}".format(100*win_count/num_simulations))
	print("Average Score: {0}".format(sum(all_scores) / float(len(all_scores))))

if __name__ == '__main__':
	main()

	
