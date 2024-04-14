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
    def __init__(self, value_iteration_model: nn.Module, gamma=0.9):
        self.value_iteration_model = value_iteration_model
        self.gamma = gamma

    def get_action(self, grid: Grid) -> torch.Tensor:
        '''Assumes model does not have a softmax layer'''
        model_outputs = self.value_iteration_model(grid.get_state())
        action = torch.softmax(model_outputs, dim=-1)
        return action

    def get_move(self, grid: Grid) -> Move:
        action = self.get_action(grid)
        move = torch.argmax(action)
        return Move(move)
    
    # Populate the Value Table
    def update_value_table(self, grid: Grid, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()
        current_state = grid.get_state()
        current_value = self.value_iteration_model(current_state)

        total_loss = 0
        for move in Move:
            # Simulate the move and get the reward
            reward = grid.process_action(move)
            next_state = grid.get_state()
            next_value = self.value_iteration_model(next_state).detach()

            # Compute the target value using the reward and discounted next value
            target_value = reward + self.gamma * torch.max(next_value)
            loss = nn.MSELoss()(current_value, target_value.unsqueeze(0))

            # Accumulate loss to optimize
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        return total_loss / len(Move)


    def value_iteration(self, grids, optimizer: torch.optim.Optimizer, num_iterations: int):
        for _ in range(num_iterations):
            total_loss = 0
            for grid in grids:  # Assuming grids is a list of Grid instances
                loss = self.update_value_table(grid, optimizer)
                total_loss += loss
            avg_loss = total_loss / len(grids)
            print(f"Average Loss: {avg_loss}")