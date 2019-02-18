import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import json
import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt
from .load import load_data, load_data_domain_sample

torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_features = 2048
num_labels = 3334
num_epochs = 50
batch_size = 50
num_batches = int(103000 / batch_size)
train_size = 100000
test_size = 3000
learning_rate = 0.01
embedding_dim = 64
hidden_dim = 64


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim).to(device),
                torch.zeros(1, 1, self.hidden_dim).to(device))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def compute_accuracy(predicted, correct, no_other=True, other_index=3333):
    assert len(predicted) == len(correct)
    if no_other:
        no_other_index = correct != other_index
        predicted = predicted[no_other_index]
        correct = correct[no_other_index]
        if len(correct) == 0:
            return 1.0

    return (predicted == correct).sum().item() / len(correct)


if __name__ == "__main__":
    inputs = np.array([], dtype=np.int64).reshape(0, 10)
    targets = np.array([], dtype=np.int64).reshape(0, 10)

    for batch_index in range(num_batches):
        print("Load batch {}".format(batch_index + 1))
        load_inputs, load_targets = load_data(batch_size, batch_index)
        inputs = np.concatenate((inputs, load_inputs), axis=0)
        targets = np.concatenate((targets, load_targets), axis=0)

    dataset = []
    for input, target in zip(inputs, targets):
        input = torch.from_numpy(np.array(input)[np.array(input) > -1])
        target = torch.from_numpy(np.array(target)[np.array(target) > -1])
        if len(input) > 0 and len(target) > 0:
            dataset.append((input, target))

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    model = LSTMTagger(embedding_dim, hidden_dim, num_features, num_labels).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_states = []
    for epoch in range(num_epochs):
        train_state = [epoch + 1]
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for batch_index, (input, target) in enumerate(train_dataset):
            input = input.to(device)
            target = target.to(device)

            model.zero_grad()

            # Clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Run forward pass.
            tag_scores = model(input)

            # Compute the loss, gradients, and update the parameters
            loss = loss_function(tag_scores, target)
            running_loss += (loss - running_loss) / (batch_index + 1)
            loss.backward()
            optimizer.step()

            # Compute the train accuracy
            acc = compute_accuracy(torch.argmax(tag_scores, dim=1), target)
            running_acc += (acc - running_acc) / (batch_index + 1)

        train_state += [running_loss.item(), running_acc]

        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        for batch_index, (input, target) in enumerate(test_dataset):
            input = input.to(device)
            target = target.to(device)

            tag_scores = model(input)

            # Compute the loss
            loss = loss_function(tag_scores, target)
            running_loss += (loss - running_loss) / (batch_index + 1)

            # Compute the validation accuracy
            acc = compute_accuracy(torch.argmax(tag_scores, dim=1), target)
            running_acc += (acc - running_acc) / (batch_index + 1)

        train_state += [running_loss.item(), running_acc]

        print(
            "[EPOCH]: {} | [TRAIN LOSS]: {:.4f} | [TRAIN ACC]: {:.4f} | [VAL LOSS]: {:.4f} | [VAL ACC]: {:.4f}".format(
                *train_state))
        train_states.append(train_state)

    model.eval()
    with torch.no_grad():
        predicted = [torch.argmax(model(input.to(device)), dim=1) for input, target in test_dataset]
        correct = [target.to(device) for input, target in test_dataset]

        predicted = torch.cat(predicted)
        correct = torch.cat(correct)

        print("Validation accuracy: {}".format(compute_accuracy(predicted, correct)))

    json.dump(train_states, open("lstm/train_states.json", "w+"), indent=4)

    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot([train_state[1] for train_state in train_states], label="train")
    plt.plot([train_state[3] for train_state in train_states], label="val")
    plt.legend(loc="upper right")

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot([train_state[2] for train_state in train_states], label="train")
    plt.plot([train_state[4] for train_state in train_states], label="val")
    plt.legend(loc="lower right")

    # Save figure
    plt.savefig("lstm/performance.png")
