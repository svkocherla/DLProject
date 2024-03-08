from simulator.cube3 import Cube3
from simulator.cube_abc import *
from utils.enums import *

# for testing cube stuff
cube = Cube3()
cube.print_cube()

cube.process_action(Moves.BACK_CCW)

cube.print_cube()