import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np
import json
import os
from operator import itemgetter
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


def compute_accuracy(predicted, correct, no_other=True, other_index=3333):
    assert len(predicted) == len(correct)
    if no_other:
        no_other_index = correct != other_index
        predicted = predicted[no_other_index]
        correct = correct[no_other_index]
        if len(correct) == 0:
            return np.nan

    return (predicted == correct).sum().item() / len(correct)


def get_labels(target):
    return [wordlist[i] for i in target if i != -1]


def predict(input, target):
    prediction = np.array([-1] * 10)

    for column_index in range((target != -1).sum()):
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
    table_path = 'data/train/4/1438042982502.13_20150728002302-00334-ip-10-236-191-2_468708662_7.json'
    data = json.load(open(table_path))
    input, target = generate_input_target(data)
    prediction = predict(input, target)
    print('Predict: {}'.format(get_labels(prediction)))
    print('Correct: {}'.format(get_labels(target)))
    print('Accuracy of the network on the test table: {:.2f}%'.format(
        100 * compute_accuracy(prediction[target != -1], target[target != -1])))
