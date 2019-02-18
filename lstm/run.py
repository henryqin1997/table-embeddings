import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
from .load import load_data, load_data_domain_sample

torch.manual_seed(1)


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
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

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

    return int((predicted == correct).sum()) / len(correct)


if __name__ == '__main__':
    inputs = np.array([], dtype=np.int64).reshape(0, 10)
    targets = np.array([], dtype=np.int64).reshape(0, 10)

    batch_size = 50
    batch_index = 0

    while batch_size * batch_index < 1000:
        print('Load batch {}'.format(batch_index))
        load_inputs, load_targets = load_data_domain_sample(batch_size, batch_index)
        inputs = np.concatenate((inputs, load_inputs), axis=0)
        targets = np.concatenate((targets, load_targets), axis=0)
        batch_index += 1

    dataset = []
    for input, target in zip(inputs, targets):
        dataset.append((torch.from_numpy(np.array(input)[np.array(input) > -1]),
                        torch.from_numpy(np.array(target)[np.array(target) > -1])))

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    training_data, testing_data = torch.utils.data.random_split(dataset, [train_size, test_size])

    # These will usually be more like 32 or 64 dimensional.
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 2048, 3334)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(300):
        print('Epoch {}'.format(epoch))
        for input, target in training_data:
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

    with torch.no_grad():
        for dataset, stage in zip([training_data, testing_data], ['Train', 'Validation']):
            predicted = [torch.argmax(model(input), dim=1) for input, target in dataset]
            correct = [target for input, target in dataset]

            predicted = torch.cat(predicted)
            correct = torch.cat(correct)

            print('{} accuracy: {}'.format(stage, calculate_accuracy(predicted, correct)))
