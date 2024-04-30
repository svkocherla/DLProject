import matplotlib.pyplot as plt

from q_learning.qlearn import QLearning
from q_learning.qnetwork import QNetwork
from simulator.game15 import Grid

def plot_accs(shuffle, res):

    grid_size = 4
    learning_rate = 0.00015
    discount_factor = 0.95
    epsilon = 0.1
    max_episodes = 100000

    # turn of verbose if you want
    env = Grid(grid_size)
    q_test = QNetwork(grid_size, learning_rate, discount_factor, epsilon)
    q_learning = QLearning(q_test, max_episodes)
    
    dir = 'q_learning/models/Qnet4x4mil'
    
    checkpoints = []
    accuracies = []
    
    env = Grid(4)
    for i in range(0, 500000, 3000):
        try:
            q_test.load_model(f"{dir}/{i}")
            accuracy = q_learning.run_tests(env, num_tests=1000, verbose=False, max_shuffle=shuffle, step_limit=100, set_shuffle=True)
            checkpoints.append(i)
            accuracies.append(accuracy)
            print(f"Episode: {i}, Acc: {accuracy}")
        except:
            break

    tups = sorted(zip(checkpoints, accuracies))
    checkpoints, accuracies = zip(*tups)

    
    # plot graph
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, accuracies, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Episode")
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{res}.png")
    plt.close()

# plot_accs(15, 'graphs/mil15')
# plot_accs(20, 'graphs/mil20')



def plot_acc_vs_shuffle(res):
    grid_size = 4
    learning_rate = 0.00015
    discount_factor = 0.95
    epsilon = 0.1
    max_episodes = 100000

    # turn of verbose if you want
    env = Grid(grid_size)
    q_test = QNetwork(grid_size, learning_rate, discount_factor, epsilon)
    q_learning = QLearning(q_test, max_episodes)
    
    dir = 'q_learning/models/Qnet4x4mil'
    q_test.load_model(f"{dir}/497000")
    
    checkpoints = []
    accuracies = []
    
    env = Grid(4)
    for i in range(1,51):
        accuracy = q_learning.run_tests(env, num_tests=1000, verbose=False, max_shuffle=i, step_limit=100, set_shuffle=True)
        checkpoints.append(i)
        accuracies.append(accuracy)
        print(f"Shuffle: {i}, Acc: {accuracy}")

    tups = sorted(zip(checkpoints, accuracies))
    checkpoints, accuracies = zip(*tups)

    
    # plot graph
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, accuracies, marker='o')
    plt.xlabel("Shuffle Distance")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Shuffle Distance")
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{res}.png")
    plt.close()

plot_acc_vs_shuffle('graphs/AccVsShuffle2')
