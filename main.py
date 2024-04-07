from simulator.game15 import *
from utils.enums import *

# for testing cube stuff
grid = Grid(4)
grid.print_grid()

grid.process_move(Moves.UP)
grid.print_grid()

grid.process_move(Moves.UP)
grid.print_grid()

grid.process_move(Moves.DOWN)
grid.print_grid()

grid.process_move(Moves.RIGHT)
grid.print_grid()

grid.process_move(Moves.LEFT)
grid.print_grid()

grid.process_move(Moves.LEFT)
grid.print_grid()

grid.process_move(Moves.RIGHT)
grid.print_grid()