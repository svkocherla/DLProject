from abc import ABC, abstractmethod
import numpy as np
from utils.enums import *
import random

# cube view is like top to bottom for all side faces, away is top for top face, and closer side is top for bottom face
class Cube(ABC):
    def __init__(self, n):
        self.cube = np.zeros((6,n,n))
        self.cube[Sides.TOP.value] = np.ones((n,n)) * Colors.WHITE.value
        self.cube[Sides.BOTTOM.value] = np.ones((n,n)) * Colors.YELLOW.value
        self.cube[Sides.FRONT.value] = np.ones((n,n)) * Colors.BLUE.value
        self.cube[Sides.BACK.value] = np.ones((n,n)) * Colors.GREEN.value
        self.cube[Sides.LEFT.value] = np.ones((n,n)) * Colors.RED.value
        self.cube[Sides.RIGHT.value] = np.ones((n,n)) * Colors.ORANGE.value
        self.solved = self.cube.copy()

    @abstractmethod
    def process_action(self, move): # Only works for 2 by 2 and 3 by 3
        pass

    @abstractmethod
    def get_reward(self):
        pass

    def get_state(self):
        return self.cube

    def is_solved(self):
        return self.cube == self.solved
    
    def shuffle_n(self, n):
        # create n random moves and perform them on cube to shuffle
        actions = [Moves.LEFT_CC, Moves.LEFT_CCW, Moves.RIGHT_CC, Moves.RIGHT_CCW,
                Moves.TOP_CC, Moves.TOP_CCW, Moves.BOTTOM_CC, Moves.BOTTOM_CCW,
                Moves.FRONT_CC, Moves.FRONT_CCW, Moves.BACK_CC, Moves.BACK_CCW]
        moves = []
        for _ in range(n):
            moves.append(random.choice(actions))
        self.shuffle(moves)
    
    def shuffle(self, moves):
        # perform moves on cube to shuffle
        for move in moves:
            self.process_action(move)

    def print_cube(self):
        for side in Sides:
            print("Side: ", side)
            print(self.cube[side.value])