from q_learning.qlearn import QLearning
from q_learning.qmodel import QTabular
from q_learning.qnetwork import QNetwork
from simulator.game15 import *
from util.enums import *

if __name__ == "__main__":
    grid_size = 2
    learning_rate = 0.01
    discount_factor = 0.95
    epsilon = 0.15
    max_episodes = 10000

    # turn of verbose if you want
    env = Grid(grid_size)
    dqn = QNetwork(grid_size, learning_rate, discount_factor, epsilon)
    learning = QLearning(dqn, max_episodes)
    learning.train(env, verbose = True, shuffle_cap=20) # use shuffle cap next time
    learning.run_tests(env, num_tests=1000, verbose=True)
    dqn.save_model("q_learning/models/Qnet2x2")