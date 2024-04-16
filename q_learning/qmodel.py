from abc import ABC, abstractmethod
import json
import numpy as np
from util.enums import Move

class QModel(ABC):
    def __init__(self, grid_size, learning_rate, discount_factor, epsilon):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    @abstractmethod
    def update(self, state, action, reward, next_state):
        pass

    @abstractmethod
    def test_action(self, state):
        pass

    @abstractmethod
    def train_action(self, state):
        pass

    @abstractmethod
    def save_model(self, file):
        pass

    @abstractmethod
    def load_model(self, file):
        pass


class QTabular(QModel):
    def __init__(self, grid_size, learning_rate, discount_factor, epsilon):
        super().__init__(grid_size, learning_rate, discount_factor, epsilon)
        self.q_table = {}
    
    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)

        current_q = self.q_table[state][action.value - 1]
        max_future_q = 0
        if next_state in self.q_table:
            max_future_q = np.max(self.q_table[next_state])

        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state][action.value - 1] = new_q

    def test_action(self, state):
        if state in self.q_table:
            return Move(np.argmax(self.q_table[state]) + 1)
        else:
            return np.random.choice(list(Move))
    
    def train_action(self, state):
        if np.random.uniform() < self.epsilon or state not in self.q_table:
            return np.random.choice(list(Move))
        else:
            return Move(np.argmax(self.q_table[state]) + 1)
        
    def save_model(self, file):
        json_table = {}
        for state, q_values in self.q_table.items():
            json_table[str(state)] = list(q_values)

        with open(file, 'w') as f:
            json.dump(json_table, f)
    
    def load_model(self, file):
        with open(file, 'r') as f:
            json_table = json.load(f)

        self.q_table = {}
        for state, values in json_table.items():
            self.q_table[tuple(map(int, state[1:-1].split(', ')))] = np.array(values)