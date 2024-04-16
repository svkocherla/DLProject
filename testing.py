from q_learning.qlearn import QLearning
from q_learning.qmodel import QTabular
from simulator.game15 import *
from util.enums import *

if __name__ == "__main__":
    grid_size = 3
    learning_rate = 0.8
    discount_factor = 0.95
    epsilon = 0.1
    max_episodes = 100000

    # turn of verbose if you want
    env = Grid(grid_size)
    q_test = QTabular(grid_size, learning_rate, discount_factor, epsilon)
    q_test.load_model("q_learning/models/Model3x3")
    q_learning = QLearning(q_test, max_episodes)
    q_learning.run_tests(env, num_tests=1000, verbose=False, max_shuffle=100, step_limit=1000)
    # q_learning.test(env, verbose=True, very_verbose=True, max_shuffle=100, set_shuffle=True, step_limit=1000)