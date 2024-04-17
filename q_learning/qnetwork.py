from q_learning.qmodel import QModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from util.enums import Move

class QNetwork(QModel):
    def __init__(self, grid_size, learning_rate, discount_factor, epsilon):
        super().__init__(grid_size, learning_rate, discount_factor, epsilon)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def preprocess(self, state):
        # input is gridsize^2 size tuple
        state_tensor = np.zeros((self.grid_size * self.grid_size, self.grid_size * self.grid_size))
        for i in range(self.grid_size * self.grid_size):
            state_tensor[i, state[i]] = 1
        return torch.tensor(state_tensor, dtype=torch.float).reshape(self.grid_size * self.grid_size, self.grid_size, self.grid_size).unsqueeze(0)
        # return torch.tensor(state, dtype=torch.float)

    def update(self, state, action, reward, next_state):
        state_tensor = torch.tensor(self.preprocess(state), dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(self.preprocess(next_state), dtype=torch.float32).to(self.device)
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        current_q = q_values.squeeze(0)[action.value - 1]
        max_future_q = torch.max(next_q_values).item()
        target_q = reward + self.discount_factor * max_future_q

        loss = self.loss(current_q, torch.tensor(target_q).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(self.preprocess(state), dtype=torch.float32).to(self.device)
            q_values = self.model(state_tensor)
            action_index = torch.argmax(q_values).item()
            return Move(action_index + 1)

    def train_action(self, state, eps):
        if np.random.uniform() < eps:
            return np.random.choice(list(Move))
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(self.preprocess(state), dtype=torch.float32).to(self.device)
                q_values = self.model(state_tensor)
                action_index = torch.argmax(q_values).item()
                return Move(action_index + 1)

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file))