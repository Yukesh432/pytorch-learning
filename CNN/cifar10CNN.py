import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


# Define a CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                        shuffle=False, num_workers=2)

# Define a loss function and optimizer
# Check if GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
# Move the model and data to the chosen device
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    # Create empty lists to store the loss values
    training_losses = []
    val_accuracies = []

    # Train the model
    for epoch in range(50):  # You can increase the number of epochs
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)


            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                training_losses.append(running_loss / 2000)
                running_loss = 0.0
     # Validation accuracy tracking
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print(f'Validation accuracy for epoch {epoch + 1}: {val_accuracy:.2f}%')

    print('Finished Training')

    # Plot the training loss graph
    plt.plot(training_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Mini-Batch Iterations')
    plt.ylabel('Loss')
    plt.show()

    # You can save the trained model
    torch.save(net.state_dict(), 'cifar10_net.pth')
