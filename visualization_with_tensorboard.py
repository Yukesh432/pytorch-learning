import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(13)

#transforms
transform= transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5))]
)

#datasets
training_data= torchvision.datasets.FashionMNIST('./data', download=True, train= True, transform= transform)
testing_data= torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

#dataloaders
train_loader= DataLoader(training_data, batch_size= 4, shuffle= True, num_workers= 2)
test_loader= DataLoader(testing_data, batch_size=4, shuffle=False, num_workers=2)

#Output labels for classes
classes= ('T-shirt/Top', 'Trouser', 'Pullover', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag')

#helper function to show an image
def matplotlib_image(img, one_channel= False):
    if one_channel:
        img= img.mean(dim=0)
    img= img / 2+ 0.5   #unnormalize.....why??
    npimg= img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap= "Greys")
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))

        


