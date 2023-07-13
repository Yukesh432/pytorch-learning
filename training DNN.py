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
        print("100" * '--')
        #get the intermediate output using hidden layer
        out= self.linear1(xb)

        #apply the activation function
        out= F.relu(out)

        #get the prediction using output layer
        out= self.linear2(out)
        return out
    
    #prediction for training data
    def training_step(self, batch):
        images, labels= batch
        out= self(images)   #generate prediction
        loss= F.cross_entropy(out, labels)   #calculate loss
        return loss
    
    #prediction for test/validation data
    def validation_step(self, batch):
        images, labels= batch
        out= self(images)
        loss= F.cross_entropy(out, labels)
        acc= accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses= [x['val_loss'] for x in outputs]
        epoch_loss= torch.stack(batch_losses).mean()   #combine losses
        batch_accs= [x['val_acc'] for x in outputs]
        epoch_acc= torch.stack(batch_accs).mean()     #commbine accuracy
        return {'validation losss': epoch_loss.item(), 'validation accuracyy': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

    
    
def accuarcy(outputs, labels):
    _, preds= torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds== labels).item()/ len(preds))


input_size= 784
hidden_size= 32
num_classes= 10

model= MnistModel(input_size, hidden_size, output_size=num_classes)

print(model)
for t in model.parameters():
    print(t.shape)


#Data prepatation




#traning model on GPU
def get_default_device():
    '''pick gpu if available else CPU'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
device= get_default_device()
print(device)

# To move the data to device we create a helper function.
def to_device(data, device):
    """Move tensor to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking= True)

for images, labels in train_loader:
    print(images.shape)
    images= to_device(images, device)
    print(images.device)
    break

