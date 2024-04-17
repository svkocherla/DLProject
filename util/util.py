# random things that might be useful
import numpy as np
import torch

def oneHotEncode(value, size):
    arr = np.zeros(size)
    arr[value] = 1
    return arr

def one_hot_encode(grid):
    n = grid.shape[0]  # Assuming grid is n x n
    num_classes = n*n
    one_hot = torch.zeros((num_classes, n, n))
    for i in range(n):
        for j in range(n):
            one_hot[grid[i, j], i, j] = 1
    return one_hot