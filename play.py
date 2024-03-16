from connect4 import *

def main():
    """
    Play Connect Four games and evaluate the Q-learning agent.
    """
    game = Game()
    game.print_state()

    player1 = game.players[0]
    player2 = game.players[1]

    win_counts = [0, 0, 0]  # [player1 wins, player2 wins, ties]

    num_games = 200
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}")
        while not game.finished:
            game.next_move()

        game.print_state()

        if game.winner is None:
            win_counts[2] += 1
        elif game.winner == player1:
            win_counts[0] += 1
        else:
            win_counts[1] += 1

        game.new_game()

    print_stats(player1, player2, win_counts)

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