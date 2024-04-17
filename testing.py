from q_learning.qlearn import QLearning
from q_learning.qmodel import QTabular
from q_learning.qnetwork import QNetwork
from simulator.game15 import *
from util.enums import *

if __name__ == "__main__":
    grid_size = 3
    learning_rate = 0.01
    discount_factor = 0.95
    epsilon = 0.1
    max_episodes = 10000

    # turn of verbose if you want
    env = Grid(grid_size)
    q_test = QNetwork(grid_size, learning_rate, discount_factor, epsilon)
    q_test.load_model("q_learning/models/Qnet3x3")
    q_learning = QLearning(q_test, max_episodes)
    q_learning.run_tests(env, num_tests=100, verbose=False, max_shuffle=30, step_limit=100)
    # q_learning.test(env, verbose=True, very_verbose=True, max_shuffle=100, set_shuffle=True, step_limit=1000)