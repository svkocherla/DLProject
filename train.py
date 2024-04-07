from models import FFN
from simulator.game15 import Grid
from torch.optim import Adam
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
optimizer = Adam(model.parameters)

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