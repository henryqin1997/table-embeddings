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
from etl.standalone import generate_input_target

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
        return self.inputs[index], self.targets[index]


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


def predict(input, target):
    prediction = np.array([-1] * 10)

    for column_index in range((target != -1).sum()):
        print('Predicting column {} ...'.format(column_index))

        model = NeuralNet().to(device)

        model.load_state_dict(torch.load('nn/model_{}.pt'.format(column_index),
                                         map_location=lambda storage, location: storage))
        model.eval()

        with torch.no_grad():
            columns = torch.from_numpy(input.reshape(-1)).float().to(device)
            out = model(columns)
            _, predicted = torch.max(out.data, 0)
            prediction[column_index] = predicted.item()

    return prediction


if __name__ == "__main__":
    data = json.load(
        open('data/train_100_sample/0/1438042988061.16_20150728002308-00106-ip-10-236-191-2_173137181_0.json'))
    input, target = generate_input_target(data)
    prediction = predict(input, target)
    print(prediction)
    print('Accuracy of the network on the test table: {:.2f}%'.format(
        100 * compute_accuracy(prediction[target != -1], target[target != -1])))
