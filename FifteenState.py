import numpy as np

class FifteenState:
    NUM_CELLS = 16

    def __init__(self, numbers):
        self.n = int(np.sqrt(len(numbers)))
        self.numbers = np.array(numbers).reshape(self.n, self.n)
        indices = np.where(self.numbers == 0)
        self.empty_cell_index = (indices[0][0], indices[1][0])  # Correctly setting it as a tuple
        self.original = self.numbers.copy()
        self.solved = np.arange(1, self.n*self.n+1).reshape(self.n, self.n)
        self.solved[-1, -1] = 0
        self.rows_solved = self.count_rows_solved()

    def is_solved(self):
        return np.array_equal(self.numbers, self.solved)

    def mask(self, upper):
        return np.where(self.numbers > upper, 0, self.numbers)
    
    def __hash__(self):
        return hash(tuple(self.numbers.flatten()))  # Hash based on the flattened grid values

    def __eq__(self, other):
        # Equality check to ensure hashable objects compare correctly
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.array_equal(self.numbers, other.numbers)

    def generate_hash(self):
        masked = self.mask(4 * self.rows_solved)
        return ','.join(map(str, masked))

    def count_rows_solved(self):
        for i in range(0, self.NUM_CELLS, 4):
            if not np.array_equal(self.numbers[i:i+4], np.arange(1 + i, 5 + i)):
                return i // 4
        return 4

    def is_terminal(self):
        return self.rows_solved == 4

    def possible_actions(self):
        actions = []
        row, col = self.empty_cell_index
        if row > 0:
            actions.append('UP')
        if row < self.n - 1:
            actions.append('DOWN')
        if col > 0:
            actions.append('LEFT')
        if col < self.n - 1:
            actions.append('RIGHT')
        return actions

    def next_state(self, action):
        row, col = self.empty_cell_index
        new_numbers = self.numbers.copy()
        if action == 'UP':
            new_numbers[row, col], new_numbers[row - 1, col] = new_numbers[row - 1, col], new_numbers[row, col]
        elif action == 'DOWN':
            new_numbers[row, col], new_numbers[row + 1, col] = new_numbers[row + 1, col], new_numbers[row, col]
        elif action == 'LEFT':
            new_numbers[row, col], new_numbers[row, col - 1] = new_numbers[row, col - 1], new_numbers[row, col]
        elif action == 'RIGHT':
            new_numbers[row, col], new_numbers[row, col + 1] = new_numbers[row, col + 1], new_numbers[row, col]
        return FifteenState(new_numbers.flatten())

    def __str__(self):
        grid = self.numbers.reshape((4, 4))
        return '\n'.join(['|'.join(f"{num:3d}" if num != self.NUM_CELLS else "   " for num in row) for row in grid])
