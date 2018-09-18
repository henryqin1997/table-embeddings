"""The 0.1 version of the training network for Tables-Embedding Project"""

import torch.nn as nn
import torch.nn.functional as F

COLUMN_DATA_TYPES = 7  # NUMBER OF POSSIBLE TYPES OF DATA IN A COLUMN OF A TABLE
WORDLIST_LABEL_SIZE = 300  # NUMBER OF DIFFERENT LABELS FOR WHOLE TRAINING SET


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # COLUMN_DATA_TYPES*10 input, 2 internal layer, 1*10 output
        self.fc1 = nn.Linear(COLUMN_DATA_TYPES * 10, 2 * WORDLIST_LABEL_SIZE * 10)
        self.fc2 = nn.Linear(2 * WORDLIST_LABEL_SIZE * 10, WORDLIST_LABEL_SIZE * 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.normalize(self, x)
        return x

    def normalize(self, x):
        x.view(-1, WORDLIST_LABEL_SIZE)
        xn = F.normalize(x, p=2, dim=1)
        xn = xn.view(-1)
        return xn


