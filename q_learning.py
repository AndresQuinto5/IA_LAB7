import numpy as np
import random
import math
import pickle

class QLearning:
    """
    QLearning class implements the Q-learning algorithm for reinforcement learning.

    Parameters:
    - alpha (float): Learning rate.
    - gamma (float): Discount factor.
    - epsilon (float): Initial exploration rate.
    - epsilon_decay_rate (float): Epsilon decay rate.
    - alpha_decay (float): Learning rate decay.
    - num_actions (int): Number of possible actions.

    Attributes:
    - alpha (float): Learning rate.
    - gamma (float): Discount factor.
    - epsilon (float): Initial exploration rate.
    - epsilon_decay_rate (float): Epsilon decay rate.
    - alpha_decay (float): Learning rate decay.
    - num_actions (int): Number of possible actions.
    - q_table (dict): Q-table to store Q-values for state-action pairs.
    - episode (int): Current episode number.

    Methods:
    - train(state, action, reward, next_state): Updates the Q-table based on the given transition.
    - get_q_value(state, action): Returns the Q-value for the given state-action pair.
    - choose_action(state, legal_moves): Chooses an action based on the current state and legal moves.
    - update_q_table(state, action, reward, next_state): Updates the Q-value in the Q-table based on the given transition.
    - reset_episode(): Resets the episode counter.
    - save_q_table(filename): Saves the Q-table to a file.
    - load_q_table(filename): Loads the Q-table from a file.
    """

    def __init__(self, alpha, gamma, epsilon, epsilon_decay_rate, alpha_decay, num_actions):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate  # Epsilon decay rate
        self.alpha_decay = alpha_decay  # Learning rate decay
        self.num_actions = num_actions
        self.q_table = {}
        self.episode = 0

    def train(self, state, action, reward, next_state):
        """
        Updates the Q-table based on the given transition.

        Parameters:
        - state: Current state.
        - action: Action taken in the current state.
        - reward: Reward received for taking the action.
        - next_state: Next state after taking the action.
        """
        self.update_q_table(state, action, reward, next_state)

    def get_q_value(self, state, action):
        """
        Returns the Q-value for the given state-action pair.

        Parameters:
        - state: Current state.
        - action: Action taken in the current state.

        Returns:
        - q_value: Q-value for the given state-action pair.
        """
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]

    def choose_action(self, state, legal_moves):
        """
        Chooses an action based on the current state and legal moves.

        Parameters:
        - state: Current state.
        - legal_moves: List of legal moves in the current state.

        Returns:
        - action: Chosen action.
        """
        self.episode += 1
        epsilon = self.epsilon * math.pow(self.epsilon_decay_rate, self.episode)

        if np.random.uniform() < epsilon:
            # Explore: choose a random action from legal moves
            action = random.choice(legal_moves)
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = [self.get_q_value(state, a) for a in legal_moves]
            max_q_value = max(q_values)
            best_actions = [a for a, q in zip(legal_moves, q_values) if q == max_q_value]
            action = random.choice(best_actions)

        return action

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-value in the Q-table based on the given transition.

        Parameters:
        - state: Current state.
        - action: Action taken in the current state.
        - reward: Reward received for taking the action.
        - next_state: Next state after taking the action.
        """
        old_q_value = self.get_q_value(state, action)
        next_max_q_value = max([self.get_q_value(next_state, a) for a in range(self.num_actions)])
        self.alpha = self.alpha / (1 + self.episode * self.alpha_decay)
        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * next_max_q_value)
        self.q_table[(state, action)] = new_q_value

    def reset_episode(self):
        """
        Resets the episode counter.
        """
        self.episode = 0

    def save_q_table(self, filename):
        """
        Saves the Q-table to a file.

        Parameters:
        - filename: Name of the file to save the Q-table.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_q_table(self, filename):
        """
        Loads the Q-table from a file.

        Parameters:
        - filename: Name of the file to load the Q-table from.
        """
        print ("Rock and LOAD !!!")
        try:
            with open(filename, 'rb') as file:
                self.q_table = pickle.load(file)
        except FileNotFoundError:
            print("Q-table file not found. Starting with an empty Q-table.")