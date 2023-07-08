"""
Pytorch has two data primitives:
a. torch.utils.data.DataLoader----------> DataLoader wraps an iterable around the dataset to enable easy access to the samples.
b. torch.utils.data.Dataset------------->. Dataset stores the samples and their corresponding labels
"""

#loading the dataset from TorchVision

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data= datasets.FashionMNIST(
    root= "data",   #root is the path where the train/test data is stored
    train=True,     #specifies training or testing data
    download=True,  #downloads the data from the internet if it's not available at root
    transform=ToTensor()  #specifies the features and label transformation
)

testing_data= datasets.FashionMNIST(
    root= "data",
    train=False,
    download= True,
    transform=ToTensor()
)


# #Iterating and Visualizing the dataset

# #creating label map to understand the labels (0 is for Tshirt , 1 is for Trouser etc...)
# labels_map= {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }

# figure= plt.figure(figsize=(8,8))
# cols, rows= 5,3

# #iterate over images
# for i in range(1, cols*rows +1):
#     #randomly select an image index
#     sample_idx= torch.randint(len(training_data), size=(1,)).item()
#     #get the image and label at the selected index
#     img, label= training_data[sample_idx]

#     #add the image to the figure
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# #show the figure
# plt.show()

print(100* "--")


"""
Creating a custom dataset for our files
A custom dataset class must implement three functions:
a. __init__
b. __len__
c. __getitem__
"""

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    #initialize the directory containing the images, the annotations file, and both transforms 
    def __init__(self, annotations_file, img_dir, transform=None, target_transform= None):
        self.img_labels= pd.read_csv(annotations_file)
        self.img_dir= img_dir
        self.transform= transform
        self.target_transform= target_transform

    #The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.img_labels)
    
    #this function loads and returns a sample from the dataset at the given index
    def __getitem__(self, idx):
        #identifies the image location on disk
        img_path= os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # Then convert that to a tensor using read_image()
        image= read_image(img_path)
        #retrieve the corresponding label fro the csv data in self.img_labels
        label= self.img_labels.iloc[idx,1]
        #call the transform function
        if self.transform:
            image= self.transform(image)
        if self.target_transform:
            label= self.target_transform(label)
        return image, label
    
print(100* "--")

#Preparing our data for training with DataLoaders

from torch.utils.data import DataLoader

train_dataloader= DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader= DataLoader(testing_data, batch_size=64, shuffle= True)

#Iterating through the DataLoader
#a. Each iteration below returns a batch of train_features and train_labels(containing batch_size=64 features and labels respectively)
#b. Becuse we specify shuffle=True, after we iterate over all batches, the data is shuffled

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")