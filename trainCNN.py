import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_preparation import PuzzleDataset
from architectures import ConvNet

def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

def main():
    dataset = PuzzleDataset('path_to_your_data.npy')  # Adjust path as necessary
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ConvNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, criterion, epochs=10)
    torch.save(model.state_dict(), 'convnet_model.pth')

if __name__ == '__main__':
    main()
