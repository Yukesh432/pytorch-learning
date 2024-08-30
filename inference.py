import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

"""
# Define the model architecture (must match the original model)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Load the model state dictionary
model = NeuralNetwork()
model.load_state_dict(torch.load("fashion_mnist_state_dict.pth"))

# Set the model to evaluation mode
model.eval()

# Example of how to run inference
def predict(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Usage
prediction = predict('pytorch modules/data/R.jpeg')
print(f"Predicted class: {prediction}")
"""


"""2. Loading the Entire Model
In this scenario, the entire model (including the architecture) was saved. You can load it 
directly without needing to redefine the model class."""

"""# Define the model architecture (must match the original model)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
# Load the entire model
model = torch.load("fashion_mnist_entire_model.pth")

# Set the model to evaluation mode
model.eval()

# Example of how to run inference
def predict(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Usage
prediction = predict('pytorch modules/data/R.jpeg')

print(f"Predicted class: {prediction}")"""


"""
3. Loading a Checkpoint
In this scenario, you saved the model's state dictionary along with other information
 (e.g., optimizer state, epoch number) in a checkpoint. You'll need to load each component separately.
"""

# Define the model architecture (must match the original model)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Initialize the model
model = NeuralNetwork()

# Load the checkpoint
checkpoint = torch.load("fashion_mnist_checkpoint.pth")

# Restore the model state and optimizer state from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# If you are continuing training, you might want to load the optimizer state as well
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Load additional information if needed
epoch = checkpoint['epoch']
loss_fn = checkpoint['loss']  # Or use a predefined loss function

# Set the model to evaluation mode
model.eval()

# Example of how to run inference
def predict(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Usage
prediction = predict('pytorch modules/data/R.jpeg')

print(f"Predicted class: {prediction}")