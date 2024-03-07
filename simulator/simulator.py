from abc import ABC, abstractmethod
import numpy as np
from utils.enums import *
import random

class Cube(ABC):
    def __init__(self, n):
        self.cube = np.zeros((6,n,n))
        self.cube[Sides.TOP] = np.ones((n,n)) * Colors.WHITE 
        self.cube[Sides.BOTTOM] = np.ones((n,n)) * Colors.YELLOW 
        self.cube[Sides.FRONT] = np.ones((n,n)) * Colors.BLUE 
        self.cube[Sides.BACK] = np.ones((n,n)) * Colors.GREEN
        self.cube[Sides.LEFT] = np.ones((n,n)) * Colors.RED
        self.cube[Sides.RIGHT] = np.ones((n,n)) * Colors.ORANGE
        self.solved = self.cube.copy()

    def process_action(self, move):
        match move:
            case Moves.LEFT_CC:
                self.cube[Sides.LEFT] = np.rot90(self.cube[Sides.LEFT], k=-1)
                # front -> bottom -> back -> top

            case Moves.LEFT_CCW:
                self.cube[Sides.LEFT] = np.rot90(self.cube[Sides.LEFT])
                # front -> top -> back -> bottom

            case Moves.RIGHT_CC:
                self.cube[Sides.RIGHT] = np.rot90(self.cube[Sides.RIGHT], k=-1)
                # front -> top -> back -> bottom

            case Moves.RIGHT_CCW:
                self.cube[Sides.RIGHT] = np.rot90(self.cube[Sides.RIGHT])
                # front -> bottom -> back -> top

            case Moves.TOP_CC:
                self.cube[Sides.TOP] = np.rot90(self.cube[Sides.TOP], k=-1)
                # front -> right -> back -> left

            case Moves.TOP_CCW:
                self.cube[Sides.TOP] = np.rot90(self.cube[Sides.TOP])
                # front -> left -> back -> right

            case Moves.BOTTOM_CC:
                self.cube[Sides.BOTTOM] = np.rot90(self.cube[Sides.BOTTOM], k=-1)
                # front -> left -> back -> right

            case Moves.BOTTOM_CCW:
                self.cube[Sides.BOTTOM] = np.rot90(self.cube[Sides.BOTTOM])
                # front -> right -> back -> left

            case Moves.FRONT_CC:
                self.cube[Sides.FRONT] = np.rot90(self.cube[Sides.FRONT], k=-1)
                # top -> right -> bottom -> left

            case Moves.FRONT_CCW:
                self.cube[Sides.FRONT] = np.rot90(self.cube[Sides.FRONT])
                # top -> left -> bottom -> right

            case Moves.BACK_CC:
                self.cube[Sides.BACK] = np.rot90(self.cube[Sides.BACK], k=-1)
                # top -> left -> bottom -> right

            case Moves.BACK_CCW:
                self.cube[Sides.BACK] = np.rot90(self.cube[Sides.BACK])
                # top -> right -> bottom -> left

            case _:
                raise ValueError("invalid move")

    def get_state(self):
        return self.cube

    @abstractmethod
    def get_reward(self):
        pass

    def is_solved(self):
        return self.cube == self.solved
    
    def shuffle_n(self, n):
        # create n random moves and perform them on cube to shuffle
        actions = [Moves.LEFT_CC, Moves.LEFT_CCW, Moves.RIGHT_CC, Moves.RIGHT_CCW,
                Moves.TOP_CC, Moves.TOP_CCW, Moves.BOTTOM_CC, Moves.BOTTOM_CCW,
                Moves.FRONT_CC, Moves.FRONT_CCW, Moves.BACK_CC, Moves.BACK_CCW]
        moves = []
        for _ in range(n):
            moves.append(random.choice(moves))
        self.shuffle(moves)
    
    def shuffle(self, moves):
        # perform moves on cube to shuffle
        for move in moves:
            self.process_action(move)


class Cube2(Cube):
    def __init__(self):
        super.__init__(2)
    
    def get_reward(self):
        pass