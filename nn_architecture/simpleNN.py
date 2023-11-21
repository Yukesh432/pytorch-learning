import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class SeedNet(nn.Module):
    def __init__(self):
        super(SeedNet, self).__init__()
        # Define a simple architecture
        self.fc1 = nn.Linear(28*28, 128)  # Example for MNIST dataset
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize network and other training components
seed_net = SeedNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(seed_net.parameters(), lr=0.001)
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Train the network
for epoch in range(5):  # You can adjust the number of epochs
    for data, target in train_loader:
        optimizer.zero_grad()
        output = seed_net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Identify critical connections (heuristic approach)
# For simplicity, let's assume we're looking at the weights of the last layer
weights = seed_net.fc2.weight.data.abs().mean(dim=1)
critical_neurons = weights.argsort(descending=True)[:10]  # Top 10 neurons


class GrownNet(nn.Module):
    def __init__(self, critical_neurons):
        super(GrownNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        # Adding a new layer connected to critical neurons
        self.fc3 = nn.Linear(10, 10)
        self.critical_neurons = critical_neurons

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        critical_output = x[:, self.critical_neurons]  # Use only critical outputs
        x = self.fc3(critical_output)
        return x

grown_net = GrownNet(critical_neurons=critical_neurons)

optimizer = optim.Adam(grown_net.parameters(), lr=0.001)

# Retrain the grown network
for epoch in range(5):  # Adjust the number of epochs as needed
    for data, target in train_loader:
        optimizer.zero_grad()
        output = grown_net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# You would typically use a separate validation dataset here
# For simplicity, reusing the training data loader in this example
correct = 0
total = 0
with torch.no_grad():
    for data, target in train_loader:
        output = grown_net(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy: {100 * correct / total}%')

# Visualize the initial seed network
plt.figure(figsize=(6, 4))
plt.title("Initial Seed Network Architecture")
plt.bar(range(len(seed_net.fc2.weight[0])), seed_net.fc2.weight[0].detach().numpy())
plt.xlabel("Connections")
plt.ylabel("Weight Values")
plt.show()

# Visualize critical connections
plt.figure(figsize=(6, 4))
plt.title("Critical Connections in Seed Network")
plt.bar(range(len(weights)), weights.detach().numpy())
plt.xlabel("Neurons")
plt.ylabel("Mean Absolute Weight Value")
plt.show()

# Visualize the grown network
plt.figure(figsize=(6, 4))
plt.title("Grown Network Architecture")
plt.bar(range(len(grown_net.fc3.weight[0])), grown_net.fc3.weight[0].detach().numpy())
plt.xlabel("Connections")
plt.ylabel("Weight Values")
plt.show()