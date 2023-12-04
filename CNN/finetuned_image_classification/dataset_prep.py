import os
import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split

def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # dataset = ImageNetSubset(root_dir=data_dir, transform=transform, subset_size=subset_size)
    trainset= datasets.CIFAR10(root='./data', train=True, download=True, transform= transform)
    trainloader= DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader