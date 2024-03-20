from connect4 import *
from tqdm import tqdm

def train_qlearning_agent(game, num_episodes):
    
    """
    we are gonna use tqdm to show the progress of our training

    Trains a QLearningAgent on the Connect4 game by playing against
    a RandomPlayer opponent for a specified number of episodes.

    The QLearningAgent's experience is stored in a Q-table that maps
    game states to expected rewards for possible actions. The trained
    Q-table is saved to a file after completing all episodes.

    Args:
    game: A Connect4 game instance.
    num_episodes: Number of complete games to play against the RandomPlayer opponent.
    """

    print("Training Q-Learning Agent...")
    player1 = game.players[0]
    player2 = RandomPlayer("Random", game.colors[1])

    if player1.algorithm != "Q-Learning":
        print("Player 1 is not using the Q-Learning algorithm. Training aborted.")
        return

    player1.qlearning.episode = 0  # Reset episode count before training

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        game.new_game()
        while not game.finished:
            state = tuple(map(tuple, game.board))
            legal_moves = [col for col in range(7) if game.board[0][col] == ' ']

            if not legal_moves:
                break  # No valid moves available, skip to the next game

            action = player1.qlearning.choose_action(state, legal_moves)
            game.next_move()
            next_state = tuple(map(tuple, game.board))
            reward = game.get_reward(player1)
            player1.qlearning.train(state, action, reward, next_state)

    player1.qlearning.save_q_table('trained_q_table.pkl')
    print("Training completed.")
    print("We overwrite the new trained Q-table with the old one.")


def play_matches(game, num_games):
    """
    Play a number of matches between two players, keeping track of wins for each.

    Loads trained Q-tables if players are using Q-learning. 
    Plays full games, printing the final board each time.
    After all games, prints the win counts for each player and ties.

    Args:
    game: Game to play matches in 
    num_games: Number of games to play

    Returns:
    None
    """
    print("Playing matches...")
    player1 = game.players[0]
    player2 = game.players[1]

    if player1.algorithm == "Q-Learning":
        print("Loading Q-table for player 1...")
        player1.qlearning.load_q_table('trained_q_table.pkl')
    elif player2.algorithm == "Q-Learning":
        print("Loading Q-table for player 2...")
        player2.qlearning.load_q_table('trained_q_table.pkl')

    win_counts = [0, 0, 0]  # [player1 wins, player2 wins, ties]

    for i in range(num_games):
        print(f"Game {i+1}/{num_games}")
        game.new_game()
        
        while not game.finished:
            game.next_move()

        game.print_state()

        if game.winner is None:
            win_counts[2] += 1
        elif game.winner == player1:
            win_counts[0] += 1
        else:
            win_counts[1] += 1

    print_stats(player1, player2, win_counts)

def main():
    # Training phase
    training_game = Game()
    num_training_episodes = 30000
    train_qlearning_agent(training_game, num_training_episodes)

    # Play phase mini-max
    game = Game()
    game.print_state()
    num_games = 75
    play_matches(game, num_games)

    input("Press Enter to continue to the next set of matches...")

    # Play phase alfa-beta pruning
    game = Game()
    game.print_state()
    num_games = 75
    play_matches(game, num_games)

def print_stats(player1, player2, win_counts):
    """
    Print the game statistics in a tabular format.
    """
    total_games = sum(win_counts)
    print("\nResults after {total_games} games:")
    print("{:<20} {:<20} {:<10} {:<10}".format("Player", "Algorithm", "Wins", "Win Rate"))
    print("-" * 60)

    print("{:<20} {:<20} {:<10} {:.2f}".format(player1.name, f"{player1.type} - {player1.algorithm}", win_counts[0], win_counts[0] / total_games))
    print("{:<20} {:<20} {:<10} {:.2f}".format(player2.name, f"{player2.type} - {player2.algorithm}", win_counts[1], win_counts[1] / total_games))
    print("{:<20} {:<20} {:<10} {:.2f}".format("Ties", "-", win_counts[2], win_counts[2] / total_games))

    if player1.algorithm == "Q-Learning":
        print(f"\n{player1.name} (Q-Learning) Evaluation:")
        print(f"Win rate: {win_counts[0] / total_games:.2f}")
        print(f"Tie rate: {win_counts[2] / total_games:.2f}")
        print(f"Loss rate: {win_counts[1] / total_games:.2f}")
    elif player2.algorithm == "Q-Learning":
        print(f"\n{player2.name} (Q-Learning) Evaluation:")
        print(f"Win rate: {win_counts[1] / total_games:.2f}")
        print(f"Tie rate: {win_counts[2] / total_games:.2f}")
        print(f"Loss rate: {win_counts[0] / total_games:.2f}")
    
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()