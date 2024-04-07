from simulator.game15 import *
from utils.enums import *

# for testing cube stuff
grid = Grid(4)
grid.print_grid()

grid.process_move(Move.UP)
grid.print_grid()

grid.process_move(Move.UP)
grid.print_grid()

grid.process_move(Move.DOWN)
grid.print_grid()

grid.process_move(Move.RIGHT)
grid.print_grid()

grid.process_move(Move.LEFT)
grid.print_grid()

grid.process_move(Move.LEFT)
grid.print_grid()

grid.process_move(Move.RIGHT)
grid.print_grid()