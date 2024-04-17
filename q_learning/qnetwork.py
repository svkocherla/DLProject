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
            nn.Linear(grid_size * grid_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, len(Move))
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        print(f"device: {self.device}")

    def update(self, state, action, reward, next_state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        current_q = q_values[action.value - 1]
        max_future_q = torch.max(next_q_values).item()
        target_q = reward + self.discount_factor * max_future_q

        loss = self.loss_fn(current_q, torch.tensor(target_q).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.model(state_tensor)
            action_index = torch.argmax(q_values).item()
            return Move(action_index + 1)

    def train_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(list(Move))
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values = self.model(state_tensor)
                action_index = torch.argmax(q_values).item()
                return Move(action_index + 1)

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)

    def load_model(self, file):
        self.model.load_state_dict(torch.load(file))