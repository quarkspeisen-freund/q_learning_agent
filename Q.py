import numpy as np
from q_table_helper import q_table_helper as save_helper
from collections import deque
from time import sleep
import random


class Q():
    def __init__(self, state_space, action_space, max_mem_size, alpha, gamma, epsilon):
        assert isinstance(state_space, int)
        assert isinstance(action_space, int)
        assert isinstance(max_mem_size, int)
        assert isinstance(alpha, float)
        assert isinstance(gamma, float)
        assert isinstance(epsilon, float)

        self.action_space = action_space
        self.q_table = np.zeros([state_space, action_space])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=max_mem_size)
        super().__init__()

    def add_transition(self, state, action, reward, next_state, done):
        assert isinstance(state, int) or isinstance(state, np.int64)
        assert isinstance(reward, int) or isinstance(state, np.int64)
        assert isinstance(next_state, int) or isinstance(state, np.int64)
        assert isinstance(done, bool)

        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        for state, action, reward, next_state, done in self.memory:
            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state])

            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.q_table[state, action] = new_value

    def choose_action(self, state):
        assert isinstance(state, int) or isinstance(state, np.int64)

        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.action_space - 1)
        else:
            action = np.argmax(self.q_table[state])

        return action

    def save(self, f_name="q.txt", directory=""):
        assert isinstance(f_name, str)
        assert isinstance(directory, str)

        save_helper.save(q_table=self.q_table,f_name=directory+f_name)
    
    def load(path):
        assert isinstance(path, str)

        data = save_helper.load(path)
        return data

