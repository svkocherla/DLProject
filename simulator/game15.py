import numpy as np
from utils.enums import *
import random

class Grid():
    #init grid of size n x n; let 15 represent the empty square
    def __init__(self, n):
        self.n = n
        self.grid = np.arange(n*n).reshape(n,n)
        self.solved = np.arange(n*n).reshape(n,n)
        self.emptyLocation = (n-1,n-1)

    def process_action(self, action):
        # return reward from (old state, action) -> (new state, reward)
        oldgrid = self.grid.copy()
        self.process_move(action)
        reward = Grid.get_reward(oldgrid, self.grid.copy(), action)
        return reward


    def process_move(self, move): 
        # moves empty location in direction of move
        row = self.emptyLocation[0] 
        col = self.emptyLocation[1]

        if move == Moves.UP:
            if row == 0:
                return 'INVALID'
            self.grid[row,col], self.grid[row-1,col] = self.grid[row-1,col], self.grid[row,col]
            self.emptyLocation = (row-1,col)
            return 'VALID'

        elif move == Moves.DOWN:
            if row == self.n-1:
                return 'INVALID'
            self.grid[row,col], self.grid[row+1,col] = self.grid[row+1,col], self.grid[row,col]
            self.emptyLocation = (row+1,col)
            return 'VALID'
        
        elif move == Moves.LEFT:
            if col == 0:
                return 'INVALID'
            self.grid[row,col], self.grid[row,col-1] = self.grid[row,col-1], self.grid[row,col]
            self.emptyLocation = (row,col-1)
            return 'VALID'
        
        elif move == Moves.RIGHT:
            if col == self.n-1:
                return 'INVALID'
            self.grid[row,col], self.grid[row,col+1] = self.grid[row,col+1], self.grid[row,col]
            self.emptyLocation = (row,col+1)
            return 'VALID'
        
        else:
            return 'INVALID'


    @staticmethod
    def get_reward(self, oldgrid, newgrid, action):
        # return reward from (old state, action) -> (new state, reward)
        pass

    def get_state(self):
        return self.grid

    def is_solved(self):
        return self.grid == self.solved
    
    def shuffle_n(self, n):
        # create n random moves and perform them on grid to shuffle
        # maybe prevent back and forths??
        actions = [Moves.UP, Moves.DOWN, Moves.LEFT, Moves.RIGHT]
        moves = []
        for _ in range(n):
            moves.append(random.choice(actions))
        self.shuffle(moves)
    
    def shuffle(self, moves):
        # perform moves on grid to shuffle
        for move in moves:
            self.process_action(move)

    def print_grid(self):
        print(self.grid)