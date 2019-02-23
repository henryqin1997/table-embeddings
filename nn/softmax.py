import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
import time
from .load import load_data, load_data_100_sample

torch.manual_seed(1)


class Table(nn.Module):
    def __init__(self):
        super(Table, self).__init__()
        self.fc = nn.Sequential(
            # fc 100*300
            # relu
            # fc 300*3000
            nn.Linear(120, 300),
            nn.ReLU(),
            nn.Linear(300, 3334)
        )

    def forward(self, x):
        x = self.fc(x)


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


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0

        for i, (columns, labels) in enumerate(trainloader):
            columns = columns.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = net.forward(columns)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end - start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            columns, labels = data
            columns = columns.to(device)
            labels = labels.to(device)
            outputs = net(columns)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test columns: %d %%' % (
            100 * correct / total))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    net = Table().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.1)

    inputs, targets = load_data_100_sample()
    dataset = TableDataset(inputs, targets)

    trainset, testset = torch.utils.data.random_split(dataset, [len(dataset) - 6, 6])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                             shuffle=False)
    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)


if __name__ == "__main__":
    main()
