from simulator.game15 import *
from util.enums import *
from util.utils import loadQnetFromConfig

if __name__ == "__main__":

    # load model from config
    filename = 'Qnet4x4'
    env, dqn, train_test = loadQnetFromConfig(f'q_learning/model_configs/{filename}.json')

    # testing
    dqn.load_model(f"q_learning/models/{filename}")
    train_test.run_tests(env, num_tests=100, verbose=True, max_shuffle=10, step_limit=100, set_shuffle = True)