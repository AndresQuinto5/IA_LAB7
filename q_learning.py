import numpy as np
import random
import math

class QLearning:
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
        self.update_q_table(state, action, reward, next_state)

    def get_q_value(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]

    def choose_action(self, state, legal_moves):
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
        old_q_value = self.get_q_value(state, action)
        next_max_q_value = max([self.get_q_value(next_state, a) for a in range(self.num_actions)])
        self.alpha = self.alpha / (1 + self.episode * self.alpha_decay)
        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * next_max_q_value)
        self.q_table[(state, action)] = new_q_value

    def reset_episode(self):
        self.episode = 0