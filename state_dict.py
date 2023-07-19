import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
- state_dict is used in saving and loading the model in pytorch.
- state_dict objects are python dictionaries, tht can be saved , altered, and updated. 
- state_dict entries are used only for two things: 
        i. layers which have learable parameter in NN(eg. convolution layer, linear layer, etc)
        ii. registered buffers(batchnorm's running mean)
- optimizer object also have state_dict which stores info about optimizer state and hyperparameters

Reference: https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
"""

# Defining and intializing neural network


#defining the neural network class which inherits from nn.Module
class Net(nn.Module):
    def __init__(self):
        # calling the  constructor of the parent class i.e (nn.Module)
        super().__init__()

        # defining layer of convolutional NN, with input channel:3, output channel: 6 and kernel size: 5*5
        self.conv1= nn.Conv2d(3, 6, 5)

        # max pooling layer with kernel size of 2*2 and a stride of 2
        self.pool= nn.MaxPool2d(2,2)

        #convolutional layer with input channel:6, output channel: 16 and kernel size: 5*5
        self.conv2= nn.Conv2d(6,16,5)

        #fully connected layer with 16*5*5 input feature and 120 output feature
        self.fc1= nn.Linear(16* 5* 5, 120)

        self.fc2= nn.Linear(120, 84)

        # FC layer with 84 input and 10 output features
        self.fc3= nn.Linear(84,10)

    #Defining the forward pass of the neural network
    def forward(self,x):

        # applied the first convolution layer followed by RelU activation and max pooling
        x= self.pool(F.relu(self.conv1(x)))
        # applied 2nd convulution layer
        x= self.pool(F.relu(self.conv2(x)))
        # reshaping the tensor to a flat vector before passing to FC layer
        x= x.view(-1, 16*5*5)
        #applying the first fully connected layer, followed by Relu activaton
        x= F.relu(self.fc1(x))
        # applying the first fully connected layer, followed by Relu activaton
        x= F.relu(self.fc2(x))
        # apply the third fully connected layer (output layer)
        x= self.fc3(x)

        # return the output tensor
        return x


# creating an instance of the Net class
net= Net()

#initializing the optimizer
optimizer= optim.SGD(net.parameters(), lr= 0.001, momentum=0.6)

# NOW THAT WE HAVE CONTRUCTED OUT MODEL AND OPTIMIZER, WE CAN UNDERSTAND WHAT IS PRESERVED IN THEIR RESPECTIVE state_dict PROPERTIES

print("Model's state_dict is:")

for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    print(100*"--")
    print(net.state_dict()[param_tensor])

#print the optimizer state_dict
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


