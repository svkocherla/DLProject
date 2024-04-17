import torch
import torch.nn as nn
from simulator.game15 import Grid
import abc
import numpy as np
from reward import naive
from util.enums import Move


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, grid: Grid) -> torch.Tensor:
        '''Get model's action distribution given state'''
    
    @abc.abstractmethod
    def get_move(self, grid: Grid) -> Move:
        '''Get model's move given state'''

class NNPolicy(Policy):
    def __init__(self, model: nn.Module):
        self.model = model

    def get_action(self, grid: Grid) -> torch.Tensor:
        '''Assumes model does not have a softmax layer'''
        model_outputs = self.model(grid.get_state())
        action = torch.softmax(model_outputs)
        return action

    def get_move(self, grid: Grid) -> Move:
        action = self.get_action(grid)
        move = torch.argmax(action)
        return move
    
#NOTE: you can add Value-Iteration/Sequence-Modeling policies by subclassing Policy

# Define the Value-Iteration Policy class

class ValueIterationPolicy(Policy):
    def __init__(self, states, discount_factor=0.9, threshold=0.01):
        self.states = states
        self.values = {state.hash(): 0 if not state.is_terminal() else 100 for state in states.values()}
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.policy = {}

    def value_iteration(self):
        while True:
            delta = 0
            for state_hash, state in self.states.items():
                if state.is_terminal():
                    continue
                max_value = float('-inf')
                for action in state.possible_actions():
                    next_state = state.next_state(action)
                    reward = naive(state, action, next_state)
                    value = reward + self.discount_factor * self.values[next_state.hash()]
                    if value > max_value:
                        max_value = value
                        self.policy[state_hash] = action
                delta = max(delta, abs(self.values[state_hash] - max_value))
                self.values[state_hash] = max_value
            if delta < self.threshold:
                break

    def get_action(self, state):
        return self.get_move(state)

    def get_move(self, state):
        best_action = None
        best_value = float('-inf')
        for action in state.possible_actions():
            next_state = state.next_state(action)
            value = self.values[next_state.hash()]
            if value > best_value:
                best_value = value
                best_action = action
        return best_action


    
