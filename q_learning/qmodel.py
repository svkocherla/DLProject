from abc import ABC, abstractmethod

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