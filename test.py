# test.py
import torch
from simulator.game15 import *
from util.enums import *
from util.utils import preprocess_state
from cnn_model import CNNModel

def run_tests(model, env, num_tests=100, max_shuffle=10, step_limit=200):
    successful_tests = 0

    for _ in range(num_tests):
        env.reset()
        env.shuffle_n(max_shuffle)

        state = env.get_state()
        steps = 0
        while not env.is_solved() and steps < step_limit:
            steps += 1

            state_tensor = preprocess_state(state, model.grid_size).to(model.device)

            action_logits = model(state_tensor)
            action = torch.argmax(action_logits).item() + 1

            env.process_action(action)
            state = env.get_state()

        if env.is_solved():
            successful_tests += 1

    print(f"Success Rate: {successful_tests / num_tests * 100:.2f}%")

if __name__ == "__main__":
    grid_size = 4
    cnn_model = CNNModel(grid_size)
    cnn_model.load_state_dict(torch.load(f"cnn_model_{grid_size}x{grid_size}.pt"))
    cnn_model.to(cnn_model.device)

    env = Grid(grid_size)

    run_tests(cnn_model, env, num_tests=1000)
