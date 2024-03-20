
import os
import time
from minimax import Minimax
import random
from q_learning import QLearning
"""


Code references to make the GAME:
https://www.askpython.com/python/examples/connect-four-game

For the board and states:
https://oscarnieves100.medium.com/programming-a-connect-4-game-on-python-f0e787a3a0cf

Paper contrasting differents performance algorithms:
https://www.tandfonline.com/doi/full/10.1080/08839514.2021.1934265 
"""
class Game:
    """
    Game object that holds the state of the Connect 4 board and game values.
    """

    def __init__(self):
        # Initialize game variables
        self.round = 1
        self.finished = False
        self.winner = None
        self.turn = None
        self.players = [None, None]
        self.game_name = u"Connect four IA_LAB07"
        self.colors = ["x", "o"]
        # Randomly select the first player
        self.turn = random.choice(self.players)

        # Clear the screen and display the welcome message
        os.system(['clear', 'cls'][os.name == 'nt'])
        print(u"Welcome to {0}!".format(self.game_name))

        # Prompt for player types and create player objects
        self.create_players()

        # Randomly shuffle the players to determine who plays first
        random.shuffle(self.players)

        # Set the first player's turn
        self.turn = self.players[0]

        # Initialize the board
        self.board = [[' ' for _ in range(7)] for _ in range(6)]

    def create_players(self):
        """
        Prompts the user to choose player types (Human or Computer) and creates player objects.
        """
        for i in range(2):
            while self.players[i] is None:
                choice = input(f"Should Player {i + 1} be a Human or a Computer? Type 'H' or 'C': ").lower()
                if choice == 'h':
                    name = input(f"What is Player {i + 1}'s name? ")
                    self.players[i] = Player(name, self.colors[i])
                elif choice == 'c':
                    name = input(f"What is Player {i + 1}'s name? ")
                    algorithm = None
                    qlearning = None
                    while algorithm is None:
                        algo_choice = input(f"Select an algorithm for {name}:\n1. Minimax\n2. Alpha-Beta\n3. Q-Learning\nEnter your choice (1/2/3): ")
                        if algo_choice == "1":
                            algorithm = "Minimax"
                        elif algo_choice == "2":
                            algorithm = "Alpha-Beta"
                            difficulty = 1  # Reduce the difficulty for alpha-beta pruning
                        elif algo_choice == "3":
                            algorithm = "Q-Learning"
                            qlearning = QLearning(alpha=0.8, gamma=0.99, epsilon=1.0, epsilon_decay_rate=0.999, alpha_decay=0.001, num_actions=7)
                        else:
                            print("Invalid choice, please try again.")
                    self.players[i] = AIPlayer(name, self.colors[i], difficulty if algorithm == "Alpha-Beta" else 5, algorithm, qlearning)
                else:
                    print("Invalid choice, please try again.")
            print(f"{self.players[i].name} will be {self.colors[i]} using {self.players[i].algorithm if self.players[i].type == 'AI' else 'Human'} algorithm")

    def first_move_random(self):
        """
        Makes the first move random for Minimax and Alpha-Beta algorithms.
        """
        if self.turn.algorithm in ["Minimax", "Alpha-Beta"]:
            legal_moves = [col for col in range(7) if self.board[0][col] == ' ']
            move = random.choice(legal_moves)
            self.board[0][move] = self.turn.color
            self.switch_turn()

    def new_game(self):
        """
        Resets the game state for a new game.
        """
        self.round = 1
        self.finished = False
        self.winner = None
        self.turn = random.choice(self.players)  # Randomly select the first player
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.first_move_random()  # Make the first move random for Minimax and Alpha-Beta algorithms

        # Reset the episode count for Q-learning agents
        for player in self.players:
            if player.algorithm == "Q-Learning":
                player.qlearning.reset_episode()

    def switch_turn(self):
        """
        Switches the turn between players and increments the round.
        """
        self.turn = self.players[1] if self.turn == self.players[0] else self.players[0]
        self.round += 1

    def next_move(self):
        """
        Handles the next move in the game.
        """
        player = self.turn
        if self.round > 42:
            self.finished = True
            return

        legal_moves = [col for col in range(7) if self.board[0][col] == ' ']
        if not legal_moves:
            self.finished = True  # No valid moves available, game is finished
            return

        move = player.move(self.board)
        for i in range(6):
            if self.board[i][move] == ' ':
                self.board[i][move] = player.color
                if player.algorithm == "Q-Learning":
                    reward = self.get_reward(player)
                    player.qlearning.update_q_table(tuple(map(tuple, self.board)), move, reward, tuple(map(tuple, self.board)))
                self.switch_turn()
                self.check_for_fours()
                self.print_state()
                return

        print("Invalid move (column is full)")


    def get_reward(self, player):
        if self.winner == player:
            return 100
        elif self.winner is None:
            return 0
        else:
            return -100
    
    def check_for_fours(self):
        """
        Checks the board for any four-in-a-row and updates the game status accordingly.
        """
        for i in range(6):
            for j in range(7):
                if self.board[i][j] != ' ':
                    if self.vertical_check(i, j):
                        self.finished = True
                        return
                    if self.horizontal_check(i, j):
                        self.finished = True
                        return
                    diag_fours, _ = self.diagonal_check(i, j)
                    if diag_fours:
                        self.finished = True
                        return

    def vertical_check(self, row, col):
        """
        Checks for a vertical four-in-a-row starting at the given position.
        """
        consecutive_count = 0
        for i in range(row, 6):
            if self.board[i][col].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break

        if consecutive_count >= 4:
            self.winner = self.players[0] if self.board[row][col].lower() == self.players[0].color.lower() else self.players[1]
            return True
        return False

    def horizontal_check(self, row, col):
        """
        Checks for a horizontal four-in-a-row starting at the given position.
        """
        consecutive_count = 0
        for j in range(col, 7):
            if self.board[row][j].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break

        if consecutive_count >= 4:
            self.winner = self.players[0] if self.board[row][col].lower() == self.players[0].color.lower() else self.players[1]
            return True
        return False

    def diagonal_check(self, row, col):
        """
        Checks for a diagonal four-in-a-row (positive or negative slope) starting at the given position.
        Returns a tuple containing a boolean indicating if a four-in-a-row was found and the slope ('positive', 'negative', or 'both').
        """
        four_in_a_row = False
        slope = None

        # Check for diagonals with positive slope
        consecutive_count = 0
        j = col
        for i in range(row, 6):
            if j > 6:
                break
            elif self.board[i][j].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= 4:
            four_in_a_row = True
            slope = 'positive'
            self.winner = self.players[0] if self.board[row][col].lower() == self.players[0].color.lower() else self.players[1]

        # Check for diagonals with negative slope
        consecutive_count = 0
        j = col
        for i in range(row, -1, -1):
            if j > 6:
                break
            elif self.board[i][j].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= 4:
            four_in_a_row = True
            slope = 'negative' if slope is None else 'both'
            self.winner = self.players[0] if self.board[row][col].lower() == self.players[0].color.lower() else self.players[1]

        return four_in_a_row, slope

    def print_state(self):
        """
        Clears the screen and prints the current game state.
        """
        os.system(['clear', 'cls'][os.name == 'nt'])
        print(u"{0}!".format(self.game_name))
        print("Round: " + str(self.round))

        for i in range(5, -1, -1):
            print("\t", end="")
            for j in range(7):
                print("| " + str(self.board[i][j]), end=" ")
            print("|")
        print("\t  _   _   _   _   _   _   _ ")
        print("\t  1   2   3   4   5   6   7 ")

        if self.finished:
            print("Game Over!")
            if self.winner is not None:
                print(f"{self.winner.name} is the winner")
            else:
                print("Game was a draw")

class Player:
    """
    Player object for human players.
    """

    def __init__(self, name, color):
        self.type = "Human"
        self.name = name
        self.color = color

    def move(self, state):
        """
        Prompts the human player to enter a move (by column number).
        """
        print(f"{self.name}'s turn. {self.name} is {self.color}")
        column = None
        while column is None:
            try:
                choice = int(input("Enter a move (by column number): ")) - 1
            except ValueError:
                choice = None
            if 0 <= choice <= 6:
                column = choice
            else:
                print("Invalid choice, try again")
        return column

class AIPlayer(Player):
    """
        AIPlayer object that extends the Player class.
        The AI algorithm is minimax with optional alpha-beta pruning.
    """
    def __init__(self, name, color, difficulty=1, algorithm=None, qlearning=None):
        self.type = "AI"
        self.name = name
        self.color = color
        self.difficulty = difficulty
        self.algorithm = algorithm
        self.minimax = Minimax([])
        if algorithm == "Q-Learning":
            self.qlearning = qlearning
            self.qlearning.load_q_table('trained_q_table.pkl')  # Load the trained Q-table

    def move(self, state):
        print(f"{self.name}'s turn. {self.name} is {self.color}")
        if self.algorithm == "Q-Learning":
            legal_moves = [col for col in range(7) if self.minimax.is_legal_move(col, state)]
            action = self.qlearning.choose_action(tuple(map(tuple, state)), legal_moves)
            return action
        else:
            minimax = Minimax(state)
            if self.algorithm == "Minimax":
                best_move, _ = minimax.minimax(self.difficulty, state, self.color, False)
            elif self.algorithm == "Alpha-Beta":
                best_move, _ = minimax.minimax(self.difficulty, state, self.color, True)
            return best_move

class RandomPlayer(Player):
    '''

    this class is to emulate a player and train Q-learning

    '''
    def __init__(self, name, color):
        self.type = "Random"
        self.name = name
        self.color = color

    def move(self, state):
        legal_moves = [col for col in range(7) if state[0][col] == ' ']
        return random.choice(legal_moves)