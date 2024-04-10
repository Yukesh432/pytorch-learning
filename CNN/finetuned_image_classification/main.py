import torch.optim as optim
import torch.nn as nn
from dataset_prep import load_data  
from torch.utils.data import Dataset
from model import get_model  
from train import train  ## remains to be implemented
import torch

# Set the batch size and number of classes for your dataset and model.
batch_size = 32
num_classes = 10  

# Load the training data using a predefined function.
# It returns a DataLoader object for the training dataset.
trainloader = load_data(batch_size=batch_size)

# Initialize the model for the specified number of classes.
# The model is expected to be a modified version of a pre-trained model.
model = get_model(num_classes)

# Freeze all layers in the model to prevent their weights from being updated during training.
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final layer of the model so that its weights can be updated during training.
# This is typically done when fine-tuning a pre-trained model on a new dataset.
for param in model.fc.parameters():
    param.requires_grad = True

# Define the loss function and optimizer for training.
# CrossEntropyLoss is commonly used for classification tasks.
criterion = nn.CrossEntropyLoss()
# SGD optimizer is used here with the parameters of the final layer, a learning rate of 0.001, and momentum of 0.9.
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Train the model using the training dataset, loss function, and optimizer.
# The training function is expected to update the model weights.
train(model, trainloader, criterion, optimizer)

# Save the trained model weights to a file for later use or deployment.
torch.save(model.state_dict(), 'trained_model.pth')
