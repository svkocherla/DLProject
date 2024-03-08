from simulator.cube_abc import Cube
from utils.enums import Moves, Sides
import numpy as np

class Cube2(Cube):
    def __init__(self):
        super().__init__(2)

    def process_action(self, move):
        # note, back view is weird
        # tbh, whole thing very scuffed
        copy = self.cube.copy()
        match move:
            # verified
            case Moves.TOP_CC:
                self.cube[Sides.TOP.value] = np.rot90(self.cube[Sides.TOP.value], k=-1)
                # front -> left -> back -> right
                self.cube[Sides.FRONT.value][0,:], self.cube[Sides.LEFT.value][0,:], self.cube[Sides.BACK.value][0,:], self.cube[Sides.RIGHT.value][0,:] = \
                    copy[Sides.RIGHT.value][0,:], copy[Sides.FRONT.value][0,:], copy[Sides.LEFT.value][0,:], copy[Sides.BACK.value][0,:][-1]

            # verified
            case Moves.TOP_CCW:
                self.cube[Sides.TOP.value] = np.rot90(self.cube[Sides.TOP.value])
                # front -> right -> back -> left
                self.cube[Sides.FRONT.value][0,:], self.cube[Sides.RIGHT.value][0,:], self.cube[Sides.BACK.value][0,:], self.cube[Sides.LEFT.value][0,:] = \
                    copy[Sides.LEFT.value][0,:], copy[Sides.FRONT.value][0,:], copy[Sides.RIGHT.value][0,:], copy[Sides.BACK.value][0,:][-1]
                
            # verified
            case Moves.FRONT_CC:
                self.cube[Sides.FRONT.value] = np.rot90(self.cube[Sides.FRONT.value], k=-1)
                # top -> right -> bottom -> left
                self.cube[Sides.TOP.value][-1,:], self.cube[Sides.RIGHT.value][:,0], self.cube[Sides.BOTTOM.value][0,:], self.cube[Sides.LEFT.value][:,-1] = \
                    copy[Sides.LEFT.value][:,-1], copy[Sides.TOP.value][-1,:], copy[Sides.RIGHT.value][:,0], copy[Sides.BOTTOM.value][0,:]

            # verified
            case Moves.FRONT_CCW:
                self.cube[Sides.FRONT.value] = np.rot90(self.cube[Sides.FRONT.value])
                # top -> left -> bottom -> right
                self.cube[Sides.TOP.value][-1,:], self.cube[Sides.LEFT.value][:,-1], self.cube[Sides.BOTTOM.value][0,:], self.cube[Sides.RIGHT.value][:,0] = \
                    copy[Sides.RIGHT.value][:,0], copy[Sides.TOP.value][-1,:], copy[Sides.LEFT.value][:,-1], copy[Sides.BOTTOM.value][0,:]
                
            # verified
            case Moves.LEFT_CC:
                self.cube[Sides.LEFT.value] = np.rot90(self.cube[Sides.LEFT.value], k=-1)
                # front -> bottom -> back -> top
                self.cube[Sides.FRONT.value][:,0], self.cube[Sides.BOTTOM.value][:,0], self.cube[Sides.BACK.value][:,-1], self.cube[Sides.TOP.value][:,0] = \
                    copy[Sides.TOP.value][:,0], copy[Sides.FRONT.value][:,0], copy[Sides.BOTTOM.value][:,0], copy[Sides.BACK.value][:,-1][-1]
            
            # verified
            case Moves.LEFT_CCW:
                self.cube[Sides.LEFT.value] = np.rot90(self.cube[Sides.LEFT.value])
                # front -> top -> back -> bottom
                self.cube[Sides.FRONT.value][:,0], self.cube[Sides.TOP.value][:,0], self.cube[Sides.BACK.value][:,-1], self.cube[Sides.BOTTOM.value][:,0] = \
                    copy[Sides.BOTTOM.value][:,0], copy[Sides.FRONT.value][:,0], copy[Sides.TOP.value][:,0], copy[Sides.BACK.value][:,-1][-1]

            # verified
            case Moves.RIGHT_CC:
                self.cube[Sides.RIGHT.value] = np.rot90(self.cube[Sides.RIGHT.value], k=-1)
                # front -> top -> back -> bottom
                self.cube[Sides.FRONT.value][:,-1], self.cube[Sides.TOP.value][:,-1], self.cube[Sides.BACK.value][:,0], self.cube[Sides.BOTTOM.value][:,-1] = \
                    copy[Sides.BOTTOM.value][:,-1], copy[Sides.FRONT.value][:,-1], copy[Sides.TOP.value][:,-1], copy[Sides.BACK.value][:,0][-1]

            # verified
            case Moves.RIGHT_CCW:
                self.cube[Sides.RIGHT.value] = np.rot90(self.cube[Sides.RIGHT.value])
                # front -> bottom -> back -> top
                self.cube[Sides.FRONT.value][:,-1], self.cube[Sides.BOTTOM.value][:,-1], self.cube[Sides.BACK.value][:,0], self.cube[Sides.TOP.value][:,-1] = \
                    copy[Sides.TOP.value][:,-1], copy[Sides.FRONT.value][:,-1], copy[Sides.BOTTOM.value][:,-1], copy[Sides.BACK.value][:,0][-1]
                    
            # verified
            case Moves.BOTTOM_CC:
                self.cube[Sides.BOTTOM.value] = np.rot90(self.cube[Sides.BOTTOM.value], k=-1)
                # front -> left -> back -> right
                self.cube[Sides.FRONT.value][-1,:], self.cube[Sides.LEFT.value][-1,:], self.cube[Sides.BACK.value][-1,:], self.cube[Sides.RIGHT.value][-1,:] = \
                    copy[Sides.RIGHT.value][-1,:], copy[Sides.FRONT.value][-1,:], copy[Sides.LEFT.value][-1,:], copy[Sides.BACK.value][-1,:][-1]

            # verified
            case Moves.BOTTOM_CCW:
                self.cube[Sides.BOTTOM.value] = np.rot90(self.cube[Sides.BOTTOM.value])
                # front -> right -> back -> left
                self.cube[Sides.FRONT.value][-1,:], self.cube[Sides.RIGHT.value][-1,:], self.cube[Sides.BACK.value][-1,:], self.cube[Sides.LEFT.value][-1,:] = \
                    copy[Sides.LEFT.value][-1,:], copy[Sides.FRONT.value][-1,:], copy[Sides.RIGHT.value][-1,:], copy[Sides.BACK.value][-1,:][-1]

            # verified
            case Moves.BACK_CC:
                self.cube[Sides.BACK.value] = np.rot90(self.cube[Sides.BACK.value], k=-1)
                # top -> left -> bottom -> right
                self.cube[Sides.TOP.value][0,:], self.cube[Sides.LEFT.value][:,0], self.cube[Sides.BOTTOM.value][-1,:], self.cube[Sides.RIGHT.value][:,-1] = \
                    copy[Sides.RIGHT.value][:,-1], copy[Sides.TOP.value][0,:], copy[Sides.LEFT.value][:,0], copy[Sides.BOTTOM.value][-1,:]

            # verified
            case Moves.BACK_CCW:
                self.cube[Sides.BACK.value] = np.rot90(self.cube[Sides.BACK.value])
                # top -> right -> bottom -> left
                self.cube[Sides.TOP.value][0,:], self.cube[Sides.RIGHT.value][:,-1], self.cube[Sides.BOTTOM.value][-1,:], self.cube[Sides.LEFT.value][:,0] = \
                    copy[Sides.LEFT.value][:,0], copy[Sides.TOP.value][0,:], copy[Sides.RIGHT.value][:,-1], copy[Sides.BOTTOM.value][-1,:]

            case _:
                raise ValueError("invalid move")
    
    def get_reward(self):
        pass