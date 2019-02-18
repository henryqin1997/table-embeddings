import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


if __name__ == '__main__':
    load_inputs, load_targets = load_data_domain_sample(10, batch_index=0)
    print(load_inputs[0])
    print(load_targets[0])

    # These will usually be more like 32 or 64 dimensional.
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 2048, 3334)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(300):
        for sentence, tags in zip(load_inputs, load_targets):
            model.zero_grad()

            # Clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Get inputs ready for the network
            sentence_in = torch.from_numpy(np.array(sentence)[np.array(sentence) > -1])
            targets = torch.from_numpy(np.array(tags)[np.array(tags) > -1])

            # Run forward pass.
            tag_scores = model(sentence_in)

            # Compute the loss, gradients, and update the parameters
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        inputs = load_inputs[0]
        inputs = inputs[inputs > -1]
        tag_scores = model(torch.from_numpy(np.array(inputs)))

        print(tag_scores)
        print(torch.argmax(tag_scores, dim=1))
