import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import sys
import json
from operator import itemgetter
from .load import load_data, load_data_100_sample, load_data_domain_schemas
from .plot import plot_performance

torch.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 120
hidden_size = 500
num_labels = 3334
num_epochs = 100
batch_size = 50
learning_rate = 0.0001
test_ratio = 0.1

wordlist = list(map(itemgetter(0), json.load(open('data/wordlist_v6_index.json')).items()))
wordlist.append('OTHER')


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
        return self.inputs[index], self.targets[index], index


def compute_accuracy(predicted, correct, no_other=True, other_index=3333):
    assert len(predicted) == len(correct)
    if no_other:
        no_other_index = correct != other_index
        predicted = predicted[no_other_index]
        correct = correct[no_other_index]
        if len(correct) == 0:
            return np.nan

    return (predicted == correct).sum().item() / len(correct)


def update_stats(stats, predicted, correct, no_other=True, other_index=3333):
    assert len(predicted) == len(correct)
    if no_other:
        no_other_index = correct != other_index
        predicted = predicted[no_other_index]
        correct = correct[no_other_index]
        if len(correct) == 0:
            return

    for p, c in zip(predicted, correct):
        stats[c][p] += 1


def stats_to_dict(stats):
    h, w = stats.shape
    assert h == w
    stats_dict = {}
    for i in range(w):
        d = {wordlist[k]: int(v) for k, v in enumerate(stats[i]) if v}
        if d:
            stats_dict[wordlist[i]] = dict(sorted(d.items(), key=itemgetter(1), reverse=True))
    return stats_dict


def main():
    column_index = int(sys.argv[1])
    assert column_index in range(10)

    model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists('nn/inputs.csv') and os.path.exists('nn/targets.csv'):
        inputs = np.genfromtxt('nn/inputs.csv', dtype='int64', delimiter=',')
        targets = np.genfromtxt('nn/targets.csv', dtype='int64', delimiter=',')
    else:
        inputs, targets = load_data()
        np.savetxt('nn/inputs.csv', inputs, fmt='%i', delimiter=',')
        np.savetxt('nn/targets.csv', targets, fmt='%i', delimiter=',')

    # Validate model on the specified column index
    targets = targets.transpose()[column_index]

    # Filter dataset so that target != -1
    inputs = inputs[targets != -1]
    targets = targets[targets != -1]

    dataset = TableDataset(inputs, targets)

    test_size = int(len(dataset) * test_ratio)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model.load_state_dict(torch.load('nn/model_{}.pt'.format(column_index),
                                     map_location=lambda storage, location: storage))
    model.eval()

    print('Testing...')
    running_acc = 0.0

    # stats is a 2d-array with shape (num_correct_labels, num_predicted_labels),
    # which counts the frequency of each prediction case
    stats = np.zeros((num_labels, num_labels), dtype='int64')

    with torch.no_grad():
        for batch_index, (columns, labels, indices) in enumerate(test_loader):
            columns = columns.float().to(device)
            labels = labels.to(device)
            out = model(columns)
            _, predicted = torch.max(out.data, 1)
            acc = compute_accuracy(predicted, labels)
            acc = running_acc if np.isnan(acc) else acc
            running_acc += (acc - running_acc) / (batch_index + 1)
            update_stats(stats, predicted, labels)
    print('Accuracy of the network on the test dataset: {:.2f}%'.format(
        100 * running_acc))

    # Convert stats to dict of dict,
    # 1st level key is correct label, 2nd level key is predicted label, value is frequency
    stats_dict = stats_to_dict(stats)

    # Sort by total frequency of each correct label
    stats_by_frequency = dict(sorted(stats_dict.items(), key=lambda item: sum(item[1].values()), reverse=True))
    json.dump(stats_by_frequency, open('nn/stats_by_frequency_{}.json'.format(column_index), 'w+'), indent=4)

    # Sort by accuracy of each correct label, from high to low, then sort by frequency
    stats_by_accuracy_desc = dict(sorted(stats_dict.items(),
                                         key=lambda item: ((item[1][item[0]] if item[0] in item[1] else 0) / sum(
                                             item[1].values()), sum(item[1].values())), reverse=True))
    json.dump(stats_by_accuracy_desc, open('nn/stats_by_accuracy_desc_{}.json'.format(column_index), 'w+'), indent=4)

    # Sort by accuracy of each correct label, from low to high, then sort by frequency
    stats_by_accuracy_asc = dict(sorted(stats_dict.items(),
                                        key=lambda item: ((item[1][item[0]] if item[0] in item[1] else 0) / sum(
                                            item[1].values()), -sum(item[1].values()))))
    json.dump(stats_by_accuracy_asc, open('nn/stats_by_accuracy_asc_{}.json'.format(column_index), 'w+'), indent=4)


if __name__ == "__main__":
    main()
