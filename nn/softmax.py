import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
import time
from .load import load_data, load_data_100_sample, load_data_domain_schemas

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 120
hidden_size = 500
num_labels = 3334
num_epochs = 10
batch_size = 50
learning_rate = 0.1
test_size = 3000


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)
        assert inputs.shape[0] == targets.shape[0]
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def train(train_loader, model, criterion, optimizer, device):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0

        total_iter = len(train_loader)
        for i, (columns, labels) in enumerate(train_loader):
            columns = columns.float().to(device)
            labels = labels.to(device)

            out = model(columns)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                end = time.time()
                print('Epoch [{}/{}], Iter [{}/{}], Loss: {:.3f}, Elapsed time {:.3f}'.format(
                    epoch + 1, num_epochs, i + 1, total_iter, running_loss / 100, end - start))
                start = time.time()
                running_loss = 0.0


def test(test_loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            columns, labels = data
            columns = columns.float().to(device)
            labels = labels.to(device)
            outputs = model(columns)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test columns: {}%'.format(
        100 * correct / total))


def main():
    model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    inputs, targets = load_data()
    dataset = TableDataset(inputs, targets)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False)

    print('Training...')
    train(train_loader, model, criterion, optimizer, device)

    print('Testing...')
    test(test_loader, model, device)


if __name__ == "__main__":
    main()
