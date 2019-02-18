import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
from .load import load_data, load_data_domain_sample

torch.manual_seed(1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_features = 2048
num_labels = 3334
num_epochs = 100
batch_size = 50
num_batches = int(2000 / batch_size)
train_size = 1600
test_size = 400
learning_rate = 0.01
embedding_dim = 32
hidden_dim = 32


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


def calculate_accuracy(predicted, correct, no_other=True, other_index=3333):
    assert len(predicted) == len(correct)
    if no_other:
        no_other_index = correct != other_index
        predicted = predicted[no_other_index]
        correct = correct[no_other_index]

    return (predicted == correct).sum().item() / len(correct)


if __name__ == '__main__':
    inputs = np.array([], dtype=np.int64).reshape(0, 10)
    targets = np.array([], dtype=np.int64).reshape(0, 10)

    for batch_index in range(num_batches):
        print('Load batch {}'.format(batch_index + 1))
        load_inputs, load_targets = load_data_domain_sample(batch_size, batch_index)
        inputs = np.concatenate((inputs, load_inputs), axis=0)
        targets = np.concatenate((targets, load_targets), axis=0)

    dataset = []
    for input, target in zip(inputs, targets):
        dataset.append((torch.from_numpy(np.array(input)[np.array(input) > -1]),
                        torch.from_numpy(np.array(target)[np.array(target) > -1])))

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    model = LSTMTagger(embedding_dim, hidden_dim, num_features, num_labels).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    total_step = len(train_dataset)
    for epoch in range(num_epochs):
        for i, (input, target) in enumerate(train_dataset):
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
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    model.eval()
    with torch.no_grad():
        predicted = [torch.argmax(model(input.to(device)), dim=1) for input, target in test_dataset]
        correct = [target.to(device) for input, target in test_dataset]

        predicted = torch.cat(predicted)
        correct = torch.cat(correct)

        print('Test accuracy: {}'.format(calculate_accuracy(predicted, correct)))
