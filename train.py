from q_learning.qlearn import QLearning
from q_learning.qmodel import QTabular
from q_learning.qnetwork import QNetwork
from simulator.game15 import *
from util.enums import *

if __name__ == "__main__":
    grid_size = 4
    learning_rate = 0.00015
    discount_factor = 0.95
    epsilon = 0.1
    max_episodes = 10000000

    # turn off verbose if you want
    env = Grid(grid_size)
    dqn = QNetwork(grid_size, learning_rate, discount_factor, epsilon)
    learning = QLearning(dqn, max_episodes)
    learning.train(env, verbose = True, shuffle_cap=100, save_steps = True, file='q_learning/models/Qnet4x4R') # use shuffle cap next time
    dqn.save_model("q_learning/models/Qnet4x4R/final")
    learning.run_tests(env, num_tests=1000, verbose=False)