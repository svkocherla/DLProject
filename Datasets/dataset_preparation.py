import torch
from torch.utils.data import Dataset
import numpy as np

class PuzzleDataset(Dataset):
    def __init__(self, data_file):
        # Load dataset here
        # Assume data_file is a path to a numpy array
        # with shape (number_of_samples, n, n) where n*n-1 is the puzzle dimension
        self.data = np.load(data_file)
        self.labels = self.generate_labels(self.data)  # Need to define how to generate labels from data

    def __getitem__(self, index):
        # Get one-hot encoded grid
        grid = self.data[index]
        one_hot_grid = self.one_hot_encode(grid)
        label = self.labels[index]
        return torch.tensor(one_hot_grid, dtype=torch.float), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def one_hot_encode(grid):
        n = grid.shape[0]
        num_classes = n * n
        one_hot = np.zeros((num_classes, n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                one_hot[grid[i, j], i, j] = 1
        return one_hot

    @staticmethod
    def generate_labels(data):
        # This method should return a list or array of labels, where each label corresponds to the correct move or classification
        # Need to implement this based on how your data is structured
        # Labels could be the next move for the empty tile, encoded as integers [0,1,2,3] for [up, down, left, right]
        # Here's a dummy implementation
        return np.random.randint(0, 4, size=len(data))
