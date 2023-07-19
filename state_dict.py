import torch
import torch.nn as nn
import torch.optim as optim


"""
- state_dict is used in saving and loading the model in pytorch.
- state_dict objects are python dictionaries, tht can be saved , altered, and updated. 
- state_dict entries are used only for two things: 
        i. layers which have learable parameter in NN(eg. convolution layer, linear layer, etc)
        ii. registered buffers(batchnorm's running mean)
- optimizer object also have state_dict which stores info about optimizer state and hyperparameters
"""
