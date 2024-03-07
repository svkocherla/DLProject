import numpy as np
from utils.enums import *
from cube_abc import Cube

class Cube2(Cube):
    def __init__(self):
        super.__init__(2)

    def process_action(self, move):
        # note, back view is weird
        # tbh, whole thing very scuffed
        match move:
            case Moves.LEFT_CC:
                self.cube[Sides.LEFT] = np.rot90(self.cube[Sides.LEFT], k=-1)
                # front -> bottom -> back -> top
                self.cube[Sides.FRONT][:,0], self.cube[Sides.BOTTOM][:,0], self.cube[Sides.BACK][:,-1], self.cube[Sides.TOP][:,0] = \
                    self.cube[Sides.TOP][:,0], self.cube[Sides.FRONT][:,0], self.cube[Sides.BOTTOM][:,0], self.cube[Sides.BACK][:,-1]

            case Moves.LEFT_CCW:
                self.cube[Sides.LEFT] = np.rot90(self.cube[Sides.LEFT])
                # front -> top -> back -> bottom
                self.cube[Sides.FRONT][:,0], self.cube[Sides.TOP][:,0], self.cube[Sides.BACK][:,-1], self.cube[Sides.BOTTOM][:,0] = \
                    self.cube[Sides.BOTTOM][:,0], self.cube[Sides.FRONT][:,0], self.cube[Sides.TOP][:,0], self.cube[Sides.BACK][:,-1]

            case Moves.RIGHT_CC:
                self.cube[Sides.RIGHT] = np.rot90(self.cube[Sides.RIGHT], k=-1)
                # front -> top -> back -> bottom
                self.cube[Sides.FRONT][:,-1], self.cube[Sides.TOP][:,-1], self.cube[Sides.BACK][:,0], self.cube[Sides.BOTTOM][:,-1] = \
                    self.cube[Sides.BOTTOM][:,-1], self.cube[Sides.FRONT][:,-1], self.cube[Sides.TOP][:,-1], self.cube[Sides.BACK][:,0]

            case Moves.RIGHT_CCW:
                self.cube[Sides.RIGHT] = np.rot90(self.cube[Sides.RIGHT])
                # front -> bottom -> back -> top
                self.cube[Sides.FRONT][:,-1], self.cube[Sides.BOTTOM][:,-1], self.cube[Sides.BACK][:,0], self.cube[Sides.TOP][:,-1] = \
                    self.cube[Sides.TOP][:,-1], self.cube[Sides.FRONT][:,-1], self.cube[Sides.BOTTOM][:,-1], self.cube[Sides.BACK][:,0]

            case Moves.TOP_CC:
                self.cube[Sides.TOP] = np.rot90(self.cube[Sides.TOP], k=-1)
                # front -> right -> back -> left
                self.cube[Sides.FRONT][0,:], self.cube[Sides.RIGHT][0,:], self.cube[Sides.BACK][0,:], self.cube[Sides.LEFT][0,:] = \
                    self.cube[Sides.LEFT][0,:], self.cube[Sides.FRONT][0,:], self.cube[Sides.RIGHT][0,:], self.cube[Sides.BACK][0,:]

            case Moves.TOP_CCW:
                self.cube[Sides.TOP] = np.rot90(self.cube[Sides.TOP])
                # front -> left -> back -> right
                self.cube[Sides.FRONT][0,:], self.cube[Sides.LEFT][0,:], self.cube[Sides.BACK][0,:], self.cube[Sides.RIGHT][0,:] = \
                    self.cube[Sides.RIGHT][0,:], self.cube[Sides.FRONT][0,:], self.cube[Sides.LEFT][0,:], self.cube[Sides.BACK][0,:]

            case Moves.BOTTOM_CC:
                self.cube[Sides.BOTTOM] = np.rot90(self.cube[Sides.BOTTOM], k=-1)
                # front -> left -> back -> right
                self.cube[Sides.FRONT][-1,:], self.cube[Sides.LEFT][-1,:], self.cube[Sides.BACK][-1,:], self.cube[Sides.RIGHT][-1,:] = \
                    self.cube[Sides.RIGHT][-1,:], self.cube[Sides.FRONT][-1,:], self.cube[Sides.LEFT][-1,:], self.cube[Sides.BACK][-1,:]

            case Moves.BOTTOM_CCW:
                self.cube[Sides.BOTTOM] = np.rot90(self.cube[Sides.BOTTOM])
                # front -> right -> back -> left
                self.cube[Sides.FRONT][-1,:], self.cube[Sides.RIGHT][-1,:], self.cube[Sides.BACK][-1,:], self.cube[Sides.LEFT][-1,:] = \
                    self.cube[Sides.LEFT][-1,:], self.cube[Sides.FRONT][-1,:], self.cube[Sides.RIGHT][-1,:], self.cube[Sides.BACK][-1,:]

            case Moves.FRONT_CC:
                self.cube[Sides.FRONT] = np.rot90(self.cube[Sides.FRONT], k=-1)
                # top -> right -> bottom -> left
                self.cube[Sides.TOP][-1,:], self.cube[Sides.RIGHT][:,0], self.cube[Sides.BOTTOM][0,:], self.cube[Sides.LEFT][:,-1] = \
                    self.cube[Sides.LEFT][:,-1], self.cube[Sides.TOP][-1,:], self.cube[Sides.RIGHT][:,0], self.cube[Sides.BOTTOM][0,:]

            case Moves.FRONT_CCW:
                self.cube[Sides.FRONT] = np.rot90(self.cube[Sides.FRONT])
                # top -> left -> bottom -> right
                self.cube[Sides.TOP][-1,:], self.cube[Sides.LEFT][:,-1], self.cube[Sides.BOTTOM][0,:], self.cube[Sides.RIGHT][:,0] = \
                    self.cube[Sides.RIGHT][:,0], self.cube[Sides.TOP][-1,:], self.cube[Sides.LEFT][:,-1], self.cube[Sides.BOTTOM][0,:]

            case Moves.BACK_CC:
                self.cube[Sides.BACK] = np.rot90(self.cube[Sides.BACK], k=-1)
                # top -> left -> bottom -> right
                self.cube[Sides.TOP][0,:], self.cube[Sides.LEFT][:,0], self.cube[Sides.BOTTOM][-1,:], self.cube[Sides.RIGHT][:,-1] = \
                    self.cube[Sides.RIGHT][:,-1], self.cube[Sides.TOP][0,:], self.cube[Sides.LEFT][:,0], self.cube[Sides.BOTTOM][-1,:]

            case Moves.BACK_CCW:
                self.cube[Sides.BACK] = np.rot90(self.cube[Sides.BACK])
                # top -> right -> bottom -> left
                self.cube[Sides.TOP][0,:], self.cube[Sides.RIGHT][:,-1], self.cube[Sides.BOTTOM][-1,:], self.cube[Sides.LEFT][:,0] = \
                    self.cube[Sides.LEFT][:,0], self.cube[Sides.TOP][0,:],self.cube[Sides.RIGHT][:,-1], self.cube[Sides.BOTTOM][-1,:]

            case _:
                raise ValueError("invalid move")
    
    def get_reward(self):
        pass