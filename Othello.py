from copy import deepcopy

class OthelloGame:
	# Static/Class variables
	BLACK = -1
	WHITE = 1
	BOARD_SIZE = 8
	INITIAL_BOARD = [[0, 0, 0, 0, 0, 0, 0, 0],
					 [0, 0, 0, 0, 0, 0, 0, 0],
					 [0, 0, 0, 0, 0, 0, 0, 0],
					 [0, 0, 0, 1,-1, 0, 0, 0],
					 [0, 0, 0,-1, 1, 0, 0, 0],
					 [0, 0, 0, 0, 0, 0, 0, 0],
					 [0, 0, 0, 0, 0, 0, 0, 0],
					 [0, 0, 0, 0, 0, 0, 0, 0]]
	DIRECTIONS = [(-1, -1), (-1,  0), (-1, 1), (0, -1),
				  ( 0,  1), ( 1, -1), ( 1, 0), (1,  1)]

	def __init__(self, moves):
		self.turns = self.simulate_game(moves)
		self.scores = OthelloGame.compute_scores(self.turns[-1].final_state)

	def simulate_game(self, moves):
		
		turns = [OthelloTurn(OthelloGame.INITIAL_BOARD, OthelloGame.BLACK, moves[0])]

		move_index = 1
		while move_index < len(moves):
			next_board = turns[-1].final_state
			next_player = -turns[-1].player
			next_move = moves[move_index]
			
			next_turn = OthelloTurn(next_board, next_player, next_move)

			if len(next_turn.legal_moves) == 0:
				next_turn = OthelloTurn(next_board, -next_player, next_move)
			
			turns.append(next_turn)
			move_index += 1

		return turns

	@staticmethod
	def compute_scores(board):
		scores = { OthelloGame.WHITE : 0, OthelloGame.BLACK: 0, 0 : 0}
		for row in range(OthelloGame.BOARD_SIZE):
			for col in range(OthelloGame.BOARD_SIZE):
				scores[board[row][col]] += 1

		if scores[OthelloGame.WHITE] > scores[OthelloGame.BLACK]:
			scores[OthelloGame.WHITE] += scores[0]
		elif scores[OthelloGame.BLACK] > scores[OthelloGame.WHITE]:
			scores[OthelloGame.BLACK] += scores[0]
		else:
			scores[OthelloGame.WHITE] += scores[0]//2
			scores[OthelloGame.BLACK] += scores[0]//2

		del scores[0]
		return scores


class OthelloTurn:

	def __init__(self, initial_state, player, move):
		self.initial_state = initial_state
		self.legal_moves = self.get_legal_moves(initial_state, player)

		self.final_state = self.make_move(self.initial_state, player, move)
		self.player = player
		self.move = move

	@staticmethod
	def get_legal_moves(board, player):
		legal_moves = {}
		
		for row in range(OthelloGame.BOARD_SIZE):
			for col in range(OthelloGame.BOARD_SIZE):
				move = (row, col)
				flipped_pieces = OthelloTurn.get_flipped_pieces(board, player, move)
				if len(flipped_pieces) > 0:
					legal_moves[move] = flipped_pieces
		
		return legal_moves

	@staticmethod
	def get_flipped_pieces(board, player, move):
		row, col = move
		if not OthelloTurn.in_bounds(row, col) or board[row][col] != 0: return []

		flipped_pieces = []
		for r, c in OthelloGame.DIRECTIONS:
			flipped_pieces.extend(OthelloTurn.get_flipped_pieces_in_direction(board, player, row, col, r, c))

		return flipped_pieces

	@staticmethod
	def get_flipped_pieces_in_direction(board, player, row, col, r, c):
		current_row, current_col = row + r, col + c
		pieces = []
		
		while OthelloTurn.in_bounds(current_row, current_col) and board[current_row][current_col] == -player:
			pieces.append((current_row, current_col))
			current_row += r
			current_col += c
		
		if OthelloTurn.in_bounds(current_row, current_col) and board[current_row][current_col] == player and len(pieces) > 0:
			return pieces
		else:
			return []

	@staticmethod
	def in_bounds(row, col):
		return 0 <= row and row < OthelloGame.BOARD_SIZE and 0 <= col and col < OthelloGame.BOARD_SIZE

	@staticmethod
	def make_move(board, player = None, move = None):
		assert OthelloTurn.in_bounds(move[0], move[1])
		new_board = deepcopy(board)
		
		row, col = move
		new_board[row][col] = player
		for r, c in OthelloGame.DIRECTIONS:
			for piece in OthelloTurn.get_flipped_pieces_in_direction(board, player, row, col, r, c):
				new_board[piece[0]][piece[1]] = player
		
		return new_board

	@staticmethod
	def print_board(board):
		print('   0 1 2 3 4 5 6 7 ')
		print('  +-+-+-+-+-+-+-+-+')
		for row in range(OthelloGame.BOARD_SIZE):
			print('{0} |'.format(row), end='')
			for col in range(OthelloGame.BOARD_SIZE):
				if board[row][col] == OthelloGame.BLACK:
					char = 'B'
				elif board[row][col] == OthelloGame.WHITE:
					char = 'W'
				else:
					char = ' '
				print(char, end='|')
			print('\n  +-+-+-+-+-+-+-+-+')