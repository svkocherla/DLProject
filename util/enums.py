from enum import Enum, auto

# Actions that agent can take
class Move(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

def oppositeMove(move):
    if move == Move.UP:
        return Move.DOWN
    if move == Move.DOWN:
        return Move.UP
    if move == Move.LEFT:
        return Move.RIGHT
    if move == Move.RIGHT:
        return Move.LEFT