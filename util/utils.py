import json

from q_learning.qlearn import QLearning
from q_learning.qnetwork import QNetwork
from q_learning.qtabular import QTabular
from simulator.game15 import Grid
import torch.nn as nn
import numpy as np # needed for the eval


def loadQNetFromConfig(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    grid_size = config['grid_size']
    learning_rate = config['learning_rate']
    discount_factor = config['discount_factor']
    epsilon_function_str = config['epsilon_function']
    epsilon = eval(epsilon_function_str)
    max_episodes = config['max_episodes']
    model_layers = config['model_layers']
    model = build_model(model_layers)

    env = Grid(grid_size)
    dqn = QNetwork(grid_size, learning_rate, discount_factor, epsilon, model)
    return env, dqn, QLearning(dqn, max_episodes)

def loadQTableFromConfig(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    grid_size = config['grid_size']
    learning_rate = config['learning_rate']
    discount_factor = config['discount_factor']
    epsilon_function_str = config['epsilon_function']
    epsilon = eval(epsilon_function_str)
    max_episodes = config['max_episodes']

    env = Grid(grid_size)
    qtable = QTabular(grid_size, learning_rate, discount_factor, epsilon)
    return env, qtable, QLearning(qtable, max_episodes)


def build_model(model_layers):
    layer_list = []
    for layer in model_layers:
        layer_type = layer.pop('type')
        layer_list.append(getattr(nn, layer_type)(**layer))
    return nn.Sequential(*layer_list)