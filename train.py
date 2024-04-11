from models import FFN
from simulator.game15 import Grid
from util.enums import *
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser("Train RL agent on Game15")
parser.add_argument("-n", default=3, type=int, help="n x n game board")
# checkpoint file
# model_architecture
# output dir for training details
## training_loop

n = 3
max_steps=10**5
n_epochs=10
n_episodes=10

game = Grid(n)
model = FFN(n * n, 100) # not sure what game state representation we should use
optimizer = optim.Adam(model.parameters)

# TODO: we will need to implement distributed RL methods
# single node for now
for e in n_episodes:
    for s in max_steps:
        if game.is_solved():
            break
        optimizer.zero_grad()

        move = model(game)
        reward = game.process_move(move)
        loss = reward
        loss.backward()
        optimizer.step()

# TODO: save model to checkpoint


def train_reinforce(model, reward, optimizer="adam", learning_rate=.01):
    out_actions = 4

    game = Grid()

    # actions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    actions = [0, 1, 2, 3]
    opt = None
    if optimizer is "adam":
        opt = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer is "sgd":
        opt = optim.SGD(model.parameters(), lr=learning_rate)

    for _ in range(episodes):
        states = []
        action_buffer = []
        reward_buffer = []

        while not game.is_solved():
            state = game.get_state()
            action_dist = torch.softmax(model(state)) # depends on if model output is logits or values
            action = np.random.choice(actions, p = action_dist.numpy())
            action_buffer.append(action)
        
            r = reward(state, action)
            reward_buffer.append(reward)

            game.process_move(action)

        reward_buffer = torch.tensor(reward_buffer)


    for step in range(len(states)):
        future_rewards = torch.sum(reward_buffer[step:])

        state = states[step]
        action = action_buffer[step]

        log_prob = torch.log(model(state))[action]
        loss = -log_prob * future_rewards
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


def train_ppo(model, reward, optimizer="adam", learning_rate=.01):
    out_actions = 4

    game = Grid()

    # actions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    actions = [0, 1, 2, 3]
    opt = None
    if optimizer is "adam":
        opt = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer is "sgd":
        opt = optim.SGD(model.parameters(), lr=learning_rate)

    for _ in range(episodes):
        states = []
        action_buffer = []
        reward_buffer = []

        while not game.is_solved():
            state = game.get_state()
            action_dist = torch.softmax(model(state)) # depends on if model output is logits or values
            action = np.random.choice(actions, p = action_dist.numpy())
            action_buffer.append(action)
        
            r = reward(state, action)
            reward_buffer.append(reward)

            game.process_move(action)

        reward_buffer = torch.tensor(reward_buffer)


    for step in range(len(states)):
        future_rewards = torch.sum(reward_buffer[step:])

        state = states[step]
        action = action_buffer[step]

        log_prob = torch.log(model(state))[action]
        loss = -log_prob * future_rewards
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return model







    