import os
import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split

def load_data(batch_size=32):
    """
    Load and preprocess the CIFAR10 dataset for training.
    
    Parameters
    ----------
    batch_size : int, optional
        The number of samples to load per batch. The default is 32.
    
    Returns
    -------
    DataLoader
        A DataLoader object that yields batches of preprocessed images from the CIFAR10 dataset.
    
    Notes
    -----
    The function utilizes torchvision's datasets module to load the CIFAR10 dataset.
    It applies a common transformation pipeline, including converting images to PyTorch tensors
    and normalizing their pixel values. The DataLoader object handles shuffling and batching of the data.
    """
    
    # Define a series of transformations that will be applied to each image in the dataset.
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize pixel values.
    ])

    # Load the CIFAR10 training dataset, apply transformations, and download the data if it's not present.
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Create a DataLoader to efficiently load batches of the preprocessed training data, with shuffling.
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    return trainloader
