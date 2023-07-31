from io import open
import glob
import os
import unicodedata
import string

import torch
import torch.nn as nn


all_letters= string.ascii_letters + ".,;"

n_letters= len(all_letters)

print(all_letters)
print(n_letters)


#Define RNN module class

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size= hidden_size
        self.i2h= nn.Linear(n_categories +input_size + hidden_size, hidden_size)
        self.i20= nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o= nn.Linear(hidden_size + output_size, output_size)
        self.dropout= nn.Dropout(0.1)
        self.softmax= nn.LogSoftmax(dim=1)
        