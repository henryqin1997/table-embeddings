"""The 0.1 version of the training network for Tables-Embedding Project"""

import torch
import torch.nn as nn
import torch.nn.functional as F

COLUMN_DATA_TYPES = 7 #NUMBER OF POSSIBLE TYPES OF DATA IN A COLUMN OF A TABLE
WORDLIST_LABEL_SIZE = 300 #NUMBER OF DIFFERENT LABELS FOR WHOLE TRAINING SET

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #COLUMN_DATA_TYPES*10 input, 2 internal layer, 1*10 output
