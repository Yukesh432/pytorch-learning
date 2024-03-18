import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import itertools
from sklearn.metrics import confusion_matrix
import os

# Hyperparameters and model configurations
config = {
    "num_of_hidden_layers": 1,
    "num_of_hidden_units": 128,
    "learning_rate": 0.001,
    "optimizer": "Adam",  # Options: "SGD", "Adam", etc.
    "activation_function": "ReLU",  # Options: "Sigmoid", "ReLU", etc.
    "initialization_method": "default",  # Options: "xavier_uniform", "he_normal", etc.
    "dropout_percentage": None,
    "batch_size": 64,
    "epochs": 5,
    "patience": 5
}

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Define the model
class ANN(nn.Module):
    def __init__(self, config):
        super(ANN, self).__init__()
        # Assuming config contains 'num_of_hidden_layers' and 'hidden_units' as a list of units per layer
        layers = [784] + [config["num_of_hidden_units"] for _ in range(config["num_of_hidden_layers"])] + [10]

        self.model_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.model_layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Example to apply an initialization method, if specified
        if config.get("initialization_method", None) == "xavier_uniform":
            for layer in self.model_layers:
                if hasattr(layer, 'weight'):
                    nn.init.xavier_uniform_(layer.weight)
        
        self.activation_function = config.get("activation_function", "ReLU")

    def forward(self, x):
        for i, layer in enumerate(self.model_layers):
            x = layer(x)
            if i < len(self.model_layers) - 1:  # No activation on the output layer
                if self.activation_function == "ReLU":
                    x = F.relu(x)
                elif self.activation_function == "Sigmoid":
                    x = torch.sigmoid(x)
                # Include other activation functions as needed
        return x

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).tolist())
            all_targets.extend(target.view(-1).tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy, all_preds, all_targets

def save_learning_curves(train_acc, val_acc, train_loss, val_loss, filename_prefix):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot accuracy
    axs[0].plot(train_acc, label='Train Accuracy')
    axs[0].plot(val_acc, label='Validation Accuracy')
    axs[0].set_title('Accuracy vs. Number of Training Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc="best")
    
    # Plot loss
    axs[1].plot(train_loss, label='Train Loss')
    axs[1].plot(val_loss, label='Validation Loss')
    axs[1].set_title('Loss vs. Number of Training Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc="best")
    
    plt.tight_layout()
    
    # Save the figure in the "experiments" directory
    plt.savefig(os.path.join(experiments_dir, f"{filename_prefix}_learning_curves.png"))
    plt.close()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, filename='confusion_matrix.png'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Simplify normalization step

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Simplify the loop for adding text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the confusion matrix in the "experiments" directory
    experiments_dir = "experiments"  # Ensure this matches your directory setup
    plt.savefig(os.path.join(experiments_dir, filename))
    plt.close()

def log_metrics(train_loss, train_accuracy, val_loss, val_accuracy, early_stopping_epoch, model_training_time, config, filename="training_log.csv"):
    # Adjust the filename to include the "experiments" directory
    filename = os.path.join(experiments_dir, filename)
    with open(filename, mode='a') as csv_file:
        fieldnames = ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'early_stopping_epoch', 'model_training_time', 'config']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'early_stopping_epoch': early_stopping_epoch,
            'model_training_time': model_training_time,
            'config': str(config)
        })

def get_optimizer(model_parameters, config):
    optimizers = {
        "SGD": optim.SGD(model_parameters, lr=config["learning_rate"], momentum=0.9),
        "Adam": optim.Adam(model_parameters, lr=config["learning_rate"]),
        # Add other optimizers as needed
    }
    optimizer= optimizers.get(config["optimizer"], optim.Adam(model_parameters, lr=config["learning_rate"]))
    return optimizer
   
def config_to_string(config):
    config_items = ["{}_{}".format(key, value) for key, value in config.items()]
    return "_".join(config_items)

def main(config):
    # Transformations and DataLoader setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, loss criterion, and optimizer setup
    model = ANN(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])  # Choose optimizer based on config
    # optimizer = get_optimizer(model.parameters(), config)

    early_stopping = EarlyStopping(patience=config["patience"], verbose=True)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    start_time = time.time()
    early_stopping_epoch= None
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        val_loss, val_accuracy, _, _ = evaluate_model(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%')

        early_stopping(val_loss, model)


        if early_stopping.early_stop:
            early_stopping_epoch= epoch

            print(f"Early stopping triggered at {early_stopping_epoch} epoch!!!!!!!!!!!!!!!")
            break
    

    model_training_time = time.time() - start_time
    print(f"Training completed in {model_training_time:.2f}s")
    # Ensure early_stopping_epoch has a meaningful value for logging
    if early_stopping_epoch is None:
        early_stopping_epoch = config["epochs"]  # Indicate training went through all epochs

    
    # For logging metrics, the filename can directly use the prefix, or as defined earlier, it's already included in the function
    log_metrics(train_losses[-1], train_accuracies[-1], val_losses[-1], val_accuracies[-1], early_stopping_epoch, model_training_time, config, "training_log.csv")

    
    
    # Evaluation and plotting
    _, test_accuracy, all_preds, all_targets = evaluate_model(model, test_loader, criterion, device)
    cm = confusion_matrix(all_targets, all_preds)
    classes = list(range(10))

    # filename_prefix = f"model_cm_{config_to_string(config)}"
    # plot_confusion_matrix(confusion_matrix, classes, title='Confusion Matrix', filename= filename_prefix)

    filename_prefix = f"model_performance_{config_to_string(config)}"
    save_learning_curves(train_accuracies, val_accuracies, train_losses, val_losses, filename_prefix)


if __name__ == "__main__":
    # Ensure the "experiments" directory exists
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    main(config)