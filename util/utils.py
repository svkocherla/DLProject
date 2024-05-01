# utils.py
import numpy as np
import torch
from simulator.game15 import *
from util.enums import Move


def preprocess_state(state, grid_size):
    state_tensor = np.array(state).reshape(1, grid_size, grid_size)
    state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
    return state_tensor.unsqueeze(0)  # Adds a batch dimension

def select_best_action(env, state):
    valid_actions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    best_action = None
    min_distance = float('inf')

    state_array = np.array(state).reshape((4, 4))  # Convert to array for comparison

    for action in valid_actions:
        old_distance = np.sum(np.abs(state_array.flatten() - env.solved.flatten()))

        # Process action and retrieve new state
        action_validity = env.process_move(action)

        if action_validity == 'VALID':
            new_state = env.get_state()
            new_state_array = np.array(new_state).reshape((4, 4))
            new_distance = np.sum(np.abs(new_state_array.flatten() - env.solved.flatten()))

            distance_delta = old_distance - new_distance

            if distance_delta < min_distance:
                min_distance = distance_delta
                best_action = action

            # Reset environment to initial state
            env.set_state(state_array)

    return best_action

def generate_dataset(grid_size, num_pairs=1000):
    pairs = []

    for _ in range(num_pairs):
        env = Grid(grid_size)
        env.shuffle_n(10)
        state = env.get_state()

        best_action = select_best_action(env, state)

        pairs.append((state, best_action))

    return pairs


def train_supervised_cnn(model, dataset, epochs, learning_rate=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0

        for state, action in dataset:
            state_tensor = preprocess_state(state, model.grid_size).to(model.device)
            target_action = torch.tensor([action.value - 1], dtype=torch.long).to(model.device)

            action_logits = model(state_tensor)
            loss = criterion(action_logits, target_action)

            print(f"State: {state}")
            print(f"Predicted Action: {torch.argmax(action_logits)}")
            print(f"Target Action: {target_action}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")


def train_cnn_model(model, dataset, epochs, learning_rate=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0

        for state, action in dataset:
            state_tensor = preprocess_state(state, model.grid_size).to(model.device)

            target_action = torch.tensor([action.value - 1], dtype=torch.long).to(model.device)

            action_logits = model(state_tensor)
            loss = criterion(action_logits, target_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")




def load_cnn_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model



