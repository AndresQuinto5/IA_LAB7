from connect4 import *
from tqdm import tqdm

'''
we are gonna use tqdm to show the progress of our training
'''
def train_qlearning_agent(game, num_episodes):
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

def play_matches(game, num_games):
    print("Playing matches...")
    player1 = game.players[0]
    player2 = game.players[1]

    if player1.algorithm == "Q-Learning":
        player1.qlearning.load_q_table('trained_q_table.pkl')
    elif player2.algorithm == "Q-Learning":
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
    num_training_episodes = 20000
    train_qlearning_agent(training_game, num_training_episodes)

    # Play phase
    game = Game()
    game.print_state()

    num_games = 75
    play_matches(game, num_games)

def print_stats(player1, player2, win_counts):
    """
    Print the game statistics.
    """
    total_games = sum(win_counts)
    print(f"\nResults after {total_games} games:")
    print(f"{player1.name} ({player1.type} - {player1.algorithm}): {win_counts[0]} wins")
    print(f"{player2.name} ({player2.type} - {player2.algorithm}): {win_counts[1]} wins")
    print(f"Ties: {win_counts[2]}")

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

if __name__ == "__main__":
    main()