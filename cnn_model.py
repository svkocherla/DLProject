import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(nn.Module):
    def __init__(self, grid_size):
        super().__init__()

        self.grid_size = grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * grid_size * grid_size, 512)
        self.fc2 = nn.Linear(512, 4)  # 4 possible actions: UP, DOWN, LEFT, RIGHT

    def forward(self, state):
        # Pass through convolutional layers
        x = self.conv1(state)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = nn.ReLU()(x)

        x = self.fc2(x)
        return x  # Output 4 action logits
