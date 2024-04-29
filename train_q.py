from simulator.game15 import *
from util.enums import *
from util.utils import loadQNetFromConfig

if __name__ == "__main__":
    # load model from config
    filename = 'Qnet4x4'
    env, dqn, train_test = loadQNetFromConfig(f'q_learning/model_configs/{filename}.json')

    # training and preliminary validation
    train_test.train(env, verbose = True, shuffle_cap=20) # use shuffle cap next time
    dqn.save_model(f"q_learning/models/{filename}")

    train_test.run_tests(env, num_tests=1000, verbose=False)