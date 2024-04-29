import numpy as np
from util.enums import *
import torch

from util.enums import Move

class Grid():
    #init grid of size n x n; let 15 represent the empty square
    def __init__(self, n):
        self.n = n
        self.state = np.arange(n*n).reshape(n,n)
        self.solved = np.arange(n*n).reshape(n,n)
        self.emptyLocation = (n-1,n-1)
        self._shuffle = []

        self._move_to_direction = {
            0: np.array([-1, 0]), # UP
            1: np.array([1, 0]), # DOWN
            2: np.array([0, -1]), # LEFT
            3: np.array([0, 1]), # RIGHT
        }

    def process_move(self, move: int) -> bool: 
        # moves empty location in direction of move
        change = self._move_to_direction[move]
        newLocation = self.emptyLocation + change
        if np.any((newLocation < 0) | (newLocation >= self.n)):
            return False
        
        newLocation = tuple(newLocation)
        self.state[self.emptyLocation], self.state[newLocation] = self.state[newLocation], self.state[self.emptyLocation]
        self.emptyLocation = newLocation
        return True
    
    def process_action(self, action):
        # return reward from (old state, action) -> (new state, reward)
        old_state = self.state.copy()
        isValid = self.process_move(action)
        # if illegal move, reward is -100
        if not isValid:
            return -100
        reward = Grid.get_reward(self, old_state, action, self.state.copy())
        return reward

    @staticmethod
    def get_reward(self, old_state, action, newgrid):
        # return reward from (old state, action) -> (new state, reward)
        if self.is_solved():
            return 100
        return -1

    def get_state(self, use_ray=False):
        if use_ray:
            return dict(obs=torch.tensor(self.state))
        return tuple(self.state.flatten()) # not mutable so can be used for indexing

    def is_solved(self):
        return np.all(self.state == self.solved)

    def generate_shuffle(self, k_moves: int) -> list[int]:
        moves = np.random.choice(4, k_moves)
        return moves
    
    def clean_shuffle(self, shuffle: list) -> list[int]:
        def is_undo(prev, curr):
            return abs(prev - curr) == 1 and min(prev, curr) != 1

        stack = []
        for x in shuffle:
            if not stack or not is_undo(stack[-1], x):
                stack.append(x)
            else:
                # stack is valid and move is an undo
                stack.pop()
        return np.array(stack, dtype=np.int64)
    
    def ints_to_moves(self, move_ints):
        return [Move(i) for i in move_ints]
        
    def shuffle(self, k_moves: int):
        # generate (possibly invalid) shuffles
        moves = self.generate_shuffle(k_moves * 20)

        # remove invalid moves
        valid = [self.process_move(move) for move in moves]
        clean = self.clean_shuffle(moves[valid])[:k_moves]

        # reset board state
        self.state = self.solved.copy()
        self.emptyLocation = (self.n-1,self.n-1)

        # take the first_k valid moves
        valid = [self.process_move(move) for move in clean]
        self._shuffle = clean[valid]

    def get_effective_k(self, shuffle: list[int]) -> int:
        return len(self.clean_shuffle(shuffle))

    def print_grid(self):
        print(self.state)

    def print_shuffle(self):
        print([Move(i) for i in self._shuffle])

    # NOTE: Gymnasium env shuffles in reset as well
    def reset(self):
        self.state = self.solved.copy()
        self.emptyLocation = (self.n-1,self.n-1)
        self._shuffle = []
        
    def get_unshuffle(self, shuffle: list[int])-> list[int]:
        if len(shuffle) == 0:
            return shuffle
        flip = lambda x: x ^ 0b1
        # return flip(np.flip(self.clean_shuffle(shuffle)))
        return flip(np.flip(shuffle))