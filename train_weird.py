from simulator.game15 import *
from util.enums import *
from reinforce.reinforce import *
from architectures.weird_cnn import WeirdNet

def main():
    # load model from config
    filename = 'WeirdNet4x4'

    grid_size = 4
    learning_rate = .001
    discount_factor = .95
    max_episodes = 10000

    model = WeirdNet()
    env = Grid(grid_size)

    train_test = LearnReinforce(model, max_episodes, grid_size)
    train_test.train_reinforce(env, max_episodes=max_episodes, shuffle_cap=20, learning_rate = learning_rate, gamma = discount_factor)
    
    train_test.save_model(f"reinforce/models/WeirdNet")
    # training and preliminary validation
    train_test.run_tests(env, num_tests=1000, verbose=False)

if __name__ == "__main__":
    main()

