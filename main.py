from q_learning.qlearn import QLearning
from q_learning.qmodel import QTabular
from simulator.game15 import *
from util.enums import *

if __name__ == "__main__":
    grid_size = 3
    learning_rate = 0.8
    discount_factor = 0.95
    epsilon = 0.1
    max_episodes = 1000000

    # turn of verbose if you want
    env = Grid(grid_size)
    q_model_tabular = QTabular(grid_size, learning_rate, discount_factor, epsilon)
    q_training = QLearning(q_model_tabular, max_episodes)
    q_training.train(env, verbose = True) # use shuffle cap next time
    q_training.run_tests(env, num_tests=1000, verbose=True)
    q_model_tabular.save_model("q_learning/models/Model3x3")