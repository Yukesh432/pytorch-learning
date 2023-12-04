import torch.optim as optim
import torch.nn as nn
from dataset_prep import load_data
from torch.utils.data import Dataset
from model import get_model
from train import train
import torch


batch_size = 32
num_classes = 10  

# Load data
trainloader = load_data(batch_size=batch_size)

# Initialize model
model = get_model(num_classes)

# Freeze all layers except the final layer
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final layer
for param in model.fc.parameters():
    param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Train the model
train(model, trainloader, criterion, optimizer)

# Save the model
torch.save(model.state_dict(), 'trained_model.pth')

