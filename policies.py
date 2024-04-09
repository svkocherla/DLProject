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

# Define the Value-Iteration Policy class
class ValueIterationPolicy(Policy):
    def __init__(self, value_iteration_model: nn.Module):
        self.value_iteration_model = value_iteration_model

    def get_action(self, grid: Grid) -> torch.Tensor:
        '''Assumes model does not have a softmax layer'''
        model_outputs = self.value_iteration_model(grid.get_state())
        action = torch.softmax(model_outputs)
        return action

    def get_move(self, grid: Grid) -> Move:
        action = self.get_action(grid)
        move = torch.argmax(action)
        return move
    
    # Populate the Value Table
    def update_value_table(self, grid: Grid, targets: torch.Tensor, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()
        model_outputs = self.value_iteration_model(grid.get_state())
        loss = nn.MSELoss()(model_outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def value_iteration(self, grids, targets, optimizer: torch.optim.Optimizer, num_iterations: int):
        for _ in range(num_iterations):
            total_loss = 0
            for grid, target in zip(grids, targets):  # Assuming grids and targets are lists or batches
                loss = self.update_value_table(grid, target, optimizer)
                total_loss += loss
            avg_loss = total_loss / len(grids)
            print(f"Average Loss: {avg_loss}")

