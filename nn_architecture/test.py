# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# # Define hyperparameters
# learning_rate = 0.001
# batch_size = 64
# epochs = 500

# # Download and transform MNIST dataset
# train_dataset = datasets.MNIST(root="./data", train=True, download=True,
#                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
# test_dataset = datasets.MNIST(root="./data", train=False, download=True,
#                               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # Define the ANN model
# class ANN(nn.Module):
#     def __init__(self):
#         super(ANN, self).__init__()
        
#         self.fc1 = nn.Linear(784, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 10)
#         # torch.nn.init.kaiming_normal_(self.fc1.weight)
#         # torch.nn.init.kaiming_normal_(self.fc2.weight)
#         # torch.nn.init.kaiming_normal_(self.fc3.weight)

#     def forward(self, x):
#         x = self.fc1(x)
#         x= F.relu(x)
#         x = self.fc2(x)
#         x= F.relu(x)
#         return x


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)


# # Initialize the model, optimizer, and criterion
# model = ANN()
# #Move Model to GPU
# model = model.to(device)


# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()

# # Function to train the model
# def train(model, train_loader, optimizer, criterion):
#     model.train()
#     total_loss, total_correct, total_samples = 0, 0, 0
#     for images, labels in train_loader:
#         images = images.view(-1, 28 * 28).to(device)
#         labels= labels.to(device)
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         _, predicted = torch.max(outputs.data, 1)
#         total_correct += (predicted == labels).sum().item()
#         total_loss += loss.item()
#         total_samples += labels.size(0)
    
#     avg_loss = total_loss / len(train_loader)
#     avg_accuracy = total_correct / total_samples
#     return avg_loss, avg_accuracy

# # Function to evaluate the model
# def test(model, test_loader):
#     model.eval()
#     total_loss, total_correct, total_samples = 0, 0, 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.view(-1, 28 * 28).to(device)
#             labels= labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
            
#             _, predicted = torch.max(outputs.data, 1)
#             total_correct += (predicted == labels).sum().item()
#             total_loss += loss.item()
#             total_samples += labels.size(0)
    
#     avg_loss = total_loss / len(test_loader)
#     avg_accuracy = total_correct / total_samples
#     return avg_loss, avg_accuracy

# # Lists to store metrics
# train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

# # Training and evaluation loop
# for epoch in range(epochs):
#     train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
#     val_loss, val_accuracy = test(model, test_loader)
    
#     train_losses.append(train_loss)
#     train_accuracies.append(train_accuracy)
#     val_losses.append(val_loss)
#     val_accuracies.append(val_accuracy)
    
#     print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 500

# User-defined parameters for the model architecture
num_hidden_layers = 2  # Number of hidden layers
nodes_per_layer = 128  # Number of nodes in each hidden layer

# Download and transform MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, download=True,
                               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
test_dataset = datasets.MNIST(root="./data", train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the ANN model with a dynamic number of hidden layers
class ANN(nn.Module):
    def __init__(self, input_size, num_hidden_layers, nodes_per_layer, output_size):
        super(ANN, self).__init__()
        layers = [nn.Linear(input_size, nodes_per_layer)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
        layers.append(nn.Linear(nodes_per_layer, output_size))
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights to zeros for demonstration purposes
        for layer in self.layers:
            torch.nn.init.zeros_(layer.weight)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize the model, optimizer, and criterion with the user-defined structure
model = ANN(784, num_hidden_layers, nodes_per_layer, 10).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Function to train the model
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for images, labels in train_loader:
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_loss += loss.item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy

# Function to evaluate the model
def test(model, test_loader):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_correct / total_samples
    return avg_loss, avg_accuracy

# Lists to store metrics
train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

# Training and evaluation loop
for epoch in range(epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy = test(model, test_loader)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epoch")
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch")
plt.legend()

plt.tight_layout()
plt.show()
