import torch
from torchvision.models import resnet18, ResNet18_Weights

#load a pretrainied resnet model from torch vision
model= resnet18(weights= ResNet18_Weights.DEFAULT)

# data tensor to represent a sinfle image woth 3 channels with height=64 and width=64
# with corresponding labels initialized to some random values


data= torch.randn(1,3,64,64)

#labels in pretrained models have shape (1,1000), so ..
labels= torch.rand(1,1000)

# to make a forward pass
prediction= model(data)  #forward pass

loss= (prediction- labels).sum()

loss.backward()  #backward pass


optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step()   #gradient descent
