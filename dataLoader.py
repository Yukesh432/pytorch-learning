import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
'''
epoch= 1 forward and backward pass of ALL traning samples
batch_size= number of training samples in one forward and backward pass
number of iterations= number of passes, each pass esing [batch_size] number of samples

eg. 100 samples, batch_size= 20----> 5 iterations for 1 epoch
'''

class WineDataset(Dataset):

    def __init__(self):
        #Load the wine data from the csv file
        xy= np.loadtxt('pwine_dataset.csv', delimiter=",", dtype=np.float32, skiprows=1)

        # Convert the data to PyTorch tensors
        # `torch.from_numpy()` function converts a NumPy array to a PyTorch tensor
        self.x= torch.from_numpy(xy[:, :-1])
        self.y= torch.from_numpy(xy[:, -1:])

        # Get the number of data samples
        self.n_samples= xy.shape[0]

    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index]


    def __len__(self):
        #len(dataset)
        return self.n_samples


#Create an instance of class WineDataset
dataset= WineDataset()

""" 
* `dataset`: The dataset to be loaded in batches.
* `batch_size`: The number of samples per batch.
* `shuffle`: A boolean flag that indicates whether the data should be shuffled before each epoch.
* `num_workers`: The number of worker threads to use for loading the data."""

dataloader= DataLoader(dataset=dataset, batch_size= 5, shuffle=True, num_workers=0)

# The `dataiter` object is an iterator over the `DataLoader` object. 
# The `next()` method of the `dataiter` object returns a batch of data from the dataset.

dataiter= iter(dataloader)
data= next(dataiter)

# Print the features and labels of the data.
features, labels= data
print(features, labels)
