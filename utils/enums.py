from enum import Enum, auto

# from POV of side
# Actions that agent can take
class Moves(Enum):
    LEFT_CC = auto()
    LEFT_CCW = auto()

    RIGHT_CC = auto()
    RIGHT_CCW = auto()

    TOP_CC = auto()
    TOP_CCW = auto()

    BOTTOM_CC= auto()
    BOTTOM_CCW = auto()

    FRONT_CC = auto()
    FRONT_CCW = auto()

    BACK_CC = auto() 
    BACK_CCW = auto()

class Colors(Enum):
    WHITE = 0
    YELLOW = 1
    BLUE = 2
    GREEN = 3
    RED = 4
    ORANGE = 5


class Sides(Enum):
    TOP = 0
    BOTTOM = 1
    FRONT = 2
    BACK = 3
    LEFT = 4
    RIGHT = 5