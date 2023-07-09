"""
Reference: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


"""
First we get Device for Training
--Pytorch uses the new Metal Performance Shaders(MPS) backend for GPU trainig acceleration.
This MPS backend extends the pytorch framework, providing scripts and capabilites to set up and run operations on Mac device.
Used for Mac device only
"""
device= (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"usinf {device} device")

# Defining the class
# we define our neural network by subclassing nn.module and initialize the neural network layers in __init__.
# Every nn.Module subclasss implements the operations on input data in the forward meethod.

class NeuralNetwork(nn.Module):
    def __init__(self):
        # this super().__init__() calls the __init__() method of the parent class, which is nn.Module in this case.
        super().__init__()

        # self.flatten is an instance of nn.Flatten, which flattens the input tensor
        self.flatten= nn.Flatten()

        # it is an instance of nn.Sequential, which is a container for neural network layers.
        # the layer in the sequential container are executed in sequence
        self.linear_relu_stack= nn.Sequential(
            nn.Linear(28*28, 512),   #is a linear layer with 28*28 input features and 512 output features
            nn.ReLU(),   #applies relu function to the input tensor
            nn.Linear(512,512),   #is a linear layer with 512 input features and 512 output features
            nn.ReLU(),
            nn.Linear(512, 10),   #is a linear layer with 512 input features and 10 output features
        )

    def forward(self, x):
        x= self.flatten(x)
        logits= self.linear_relu_stack(x)
        return logits   #returns the output of the neural network
    

# We now create and instance of NeuralNetwork class and move it to the device, and print the structure

model= NeuralNetwork().to(device)

print(model)
