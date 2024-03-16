import random
'''
We use this references for our algorithms
- [MiniMax pseudo-code:](https://es.wikipedia.org/wiki/Minimax)  
- [Alpha-Beta pruning pseudo-code:](https://es.wikipedia.org/wiki/Poda_alfa-beta)  

'''
class Minimax:
    """
    Minimax object that takes a current Connect 4 board state and performs the minimax algorithm with alpha-beta pruning.
    """

    def __init__(self, board):
        self.board = [row[:] for row in board]
        self.colors = ["x", "o"]

    # def random_move(self, state, curr_player):
    #     """
    #     Selects a random legal move from the available moves.
    #     """
    #     legal_moves = [col for col in range(7) if self.is_legal_move(col, state)]
    #     if legal_moves:
    #         return random.choice(legal_moves)
    #     else:
    #         return None
        
    def minimax(self, depth, state, curr_player, use_alpha_beta, alpha=-float('inf'), beta=float('inf')):
        """
        Implements the minimax algorithm with optional alpha-beta pruning to find the best move and its associated value.
        Returns the best move (as a column number) and the associated value.
        """
        best_move = None
        best_value = -float('inf')

        # Determine the opponent's color
        opp_player = self.colors[1] if curr_player == self.colors[0] else self.colors[0]
        
        # Enumerate all legal moves
        legal_moves = [col for col in range(7) if self.is_legal_move(col, state)]

        # Base case: If the game is over or the depth is 0, return the evaluation value
        if depth == 0 or not legal_moves or self.game_is_over(state):
            return None, self.evaluate(state, curr_player)

        # Iterate over all legal moves
        # if not use_alpha_beta:
        #     print(f'{curr_player} using minimax algorithm')

        for move in legal_moves:
            new_state = self.make_move(state, move, curr_player)
            _, value = self.minimax(depth - 1, new_state, opp_player, use_alpha_beta, -beta, -alpha)
            
            if value > best_value:
                best_value = value
                best_move = move
            if use_alpha_beta:
                # print(f'{curr_player} using alpha-beta pruning')
                alpha = max(alpha, best_value)
                if beta >= alpha:
                    break
        

        return best_move, best_value


    def is_legal_move(self, column, state):
        """
        Checks if a move (column) is a legal move.
        """
        for i in range(6):
            if state[i][column] == ' ':
                return True
        return False

    def game_is_over(self, state):
        """
        Checks if the game is over by checking for any four-in-a-row.
        """
        return self.check_for_streak(state, self.colors[0], 4) >= 1 or self.check_for_streak(state, self.colors[1], 4) >= 1

    def make_move(self, state, column, color):
        """
        Creates a new board state by making a move at the specified column for the given color.
        """
        new_state = [row[:] for row in state]
        for i in range(6):
            if new_state[i][column] == ' ':
                new_state[i][column] = color
                return new_state

    def evaluate(self, state, color):
        """
        Evaluates the board state for the given color using a heuristic function.
        The heuristic is based on the number of streaks of different lengths.
        """
        opp_color = self.colors[1] if color == self.colors[0] else self.colors[0]
        my_fours = self.check_for_streak(state, color, 4)
        my_threes = self.check_for_streak(state, color, 3)
        my_twos = self.check_for_streak(state, color, 2)
        opp_fours = self.check_for_streak(state, opp_color, 4)

        if opp_fours > 0:
            return -100000
        else:
            return my_fours * 100000 + my_threes * 100 + my_twos

    def check_for_streak(self, state, color, streak):
        """
        Checks the board for streaks of the given length and color.
        Returns the total count of streaks found.
        """
        count = 0
        for i in range(6):
            for j in range(7):
                if state[i][j].lower() == color.lower():
                    count += self.vertical_streak(i, j, state, streak)
                    count += self.horizontal_streak(i, j, state, streak)
                    count += self.diagonal_streak(i, j, state, streak)
        return count

    def vertical_streak(self, row, col, state, streak):
        """
        Checks for a vertical streak of the given length starting at the specified position.
        Returns 1 if a streak is found, 0 otherwise.
        """
        consecutive_count = 0
        for i in range(row, 6):
            if state[i][col].lower() == state[row][col].lower():
                consecutive_count += 1
            else:
                break

        return 1 if consecutive_count >= streak else 0

    def horizontal_streak(self, row, col, state, streak):
        """
        Checks for a horizontal streak of the given length starting at the specified position.
        Returns 1 if a streak is found, 0 otherwise.
        """
        consecutive_count = 0
        for j in range(col, 7):
            if state[row][j].lower() == state[row][col].lower():
                consecutive_count += 1
            else:
                break

        return 1 if consecutive_count >= streak else 0

    def diagonal_streak(self, row, col, state, streak):
        """
        Checks for diagonal streaks (positive and negative slopes) of the given length starting at the specified position.
        Returns the total count of streaks found.
        """
        total = 0

        # Check for diagonals with positive slope
        consecutive_count = 0
        j = col
        for i in range(row, 6):
            if j > 6:
                break
            elif state[i][j].lower() == state[row][col].lower():
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= streak:
            total += 1

        # Check for diagonals with negative slope
        consecutive_count = 0
        j = col
        for i in range(row, -1, -1):
            if j > 6:
                break
            elif state[i][j].lower() == state[row][col].lower():
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= streak:
            total += 1

        return total
