import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import random_split

matplotlib.rcParams['figure.facecolor']= '#ffffff'


#defining the new class by extending nn.Module

class MnistModel(nn.Module):
    #feed forward NN with one hidden layer
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1= nn.Linear(input_size, hidden_size)
        self.linear2= nn.Linear(hidden_size, output_size)

    def forward(self, xb):
        #flatten the image tensors
        xb.view(xb.size(0), -1)
        #get the intermediate output using hidden layer
        out= self.linear1(xb)

        #apply the activation function
        out= F.relu(out)

        #get the prediction using output layer
        out= self.linear2(out)
        return out
    