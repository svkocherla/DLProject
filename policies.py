import torch
import torch.nn as nn
from simulator.game15 import Grid
import abc

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