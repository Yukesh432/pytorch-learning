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
    "num_of_hidden_units": 32,
    "learning_rate": 0.001,
    "optims": "SGD",  # Options: "SGD", "Adam", etc.
    "activation_function": "Gelu",  # Options: "Sigmoid", "ReLU", etc.
    "initialization_method": "he_normal",  # Options: "xavier_uniform", "he_normal", etc.
    "dropout_percentage": None,
    "batch_size": 64,
    "epochs": 50000,
    "patience": 10
}


class EarlyStopping:
    """
    Early stopping utility for halting the training process when validation loss 
    does not improve beyond a given threshold (delta) for a specified number of epochs (patience).
    
    Parameters:
    -----------
    patience : int
        The number of epochs to wait for improvement before stopping the training.
    verbose : bool
        If True, prints a message for each validation loss improvement.
    delta : float
        The minimum change in the monitored quantity to qualify as an improvement.
    
    Attributes:
    -----------
    counter : int
        Tracks the number of epochs without significant improvement.
    best_loss : float
        Records the best validation loss observed during training.
    early_stop : bool
        Indicates whether the training process should be stopped.
    """
    
    def __init__(self, patience=7, verbose=True, delta=0 ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # Initialize counter of epochs without improvement
        self.best_loss = np.Inf  # Initialize the best observed loss as infinity
        self.early_stop = False  # Initially, do not halt training
        self.delta = delta  # Set the threshold for detecting improvement
    
    def __call__(self, val_loss, model):
        """
        Call method to check if early stopping criteria are met.
        
        Parameters:
        -----------
        val_loss : float
            The current epoch's validation loss.
        model : torch.nn.Module
            The model being trained - not directly used in this simplified version,
            but can be used for saving model checkpoints.
        """
        # Check if the current validation loss shows improvement over the best loss observed
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss  # Update the best observed loss
            self.counter = 0  # Reset the improvement counter
        else:
            self.counter += 1  # Increment the counter as no improvement is observed
            # Check if the number of epochs without improvement has reached the patience limit
            if self.counter >= self.patience:
                self.early_stop = True  # Signal to stop training
            if self.verbose:
                # Optionally print the status of early stopping
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

# Define the model
class ANN(nn.Module):
    def __init__(self, config):
        super(ANN, self).__init__()
        # Assuming config contains 'num_of_hidden_layers' and 'hidden_units' as a list of units per layer
        layers = [784] + [config["num_of_hidden_units"] for _ in range(config["num_of_hidden_layers"])] + [10]
        print("*"*100)
        print(layers)

        self.model_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.model_layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # apply initialization method to each layer
        for layer in self.model_layers:
            print("LAYERS PRINTED!!!!!!!!!!!!!!")
            print(layer)
            if hasattr(layer, 'weight'):
                if config.get("initialization_method", "") == "xavier_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif config.get("initialization_method", "") == "xavier_normal":
                    nn.init.xavier_normal_(layer.weight)
                elif config.get("initialization_method", "") == "he_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                elif config.get("initialization_method", "") == "he_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity= "relu")
                elif config.get("initialization_method", "") == "ones":
                    nn.init.ones_(layer.weight)
                elif config.get("initialization_method", "") == "zeros":
                    print("Zero initialization ...............")
                    nn.init.zeros_(layer.weight)
                elif config.get("initialization_method", "") == "random_normal":
                    nn.init.normal_(layer.weight)
                elif config.get("initialization_method", "") == "random_uniform":
                    nn.init.uniform_(layer.weight)
                else:
                    print("Not a valid initialization method.........")

            print("WEIGHT INITIALIZED")
        # self.activation= config.get("activation_function", "")
        # print("ACTIVATION FUNCTION IS BEING APPLIED")

    def apply_activation(self, x):
        self.activation= config['activation_function']
        # print(self.activation)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        # Dynamically apply activation function
        if self.activation == "ReLU":
            # print("RELU FUNCTION ACTIVATEDDDDD")
            return F.relu(x)
        elif self.activation == "Sigmoid":
            return F.sigmoid(x)
        elif self.activation == "Tanh":
            return F.tanh(x)
        elif self.activation == "LeakyReLU":
            return F.leaky_relu(x)
        elif self.activation == "ELU":
            return F.elu(x)
        elif self.activation == "Softplus":
            return F.softplus(x)
        elif self.activation == "Selu":
            return F.selu(x)
        elif self.activation == "Gelu":
            return F.gelu(x)
        elif self.activation == "linear":
            return x
        else:
            print("NO such Activation function selected")
            # return x
        # Include other activation functions as needed
        # return x

    def forward(self, x):
        for _, layer in enumerate(self.model_layers[:-1]):  # Exclude the last layer for activation
            x = layer(x)
            x = self.apply_activation(x)
        x = self.model_layers[-1](x)  # Apply the last layer without activation
        return x
    


# def evaluate_model(model, test_loader, criterion, device):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     all_preds = []
#     all_targets = []
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             data = data.view(-1, 28*28)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             all_preds.extend(pred.view(-1).tolist())
#             all_targets.extend(target.view(-1).tolist())

#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
#     return test_loss, accuracy, all_preds, all_targets
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()  # Accumulate the sum of the losses
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).tolist())
            all_targets.extend(target.view(-1).tolist())

    avg_loss = running_loss / len(test_loader)  # Divide by the number of batches
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_loss, accuracy, all_preds, all_targets

def save_learning_curves(train_acc, val_acc, train_loss, val_loss, filename_prefix):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_acc, label= "Training Accuracy")
    plt.plot(val_acc, label= "Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epoch")
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()

    # Adjust the y-axis scale of the loss plot to zoom in on changes
    # axs[1].set_ylim([min(val_loss)*0.95, max(val_loss)*1.05])

    
    plt.tight_layout()
    
    # Save the figure in the "experiments" directory
    plt.savefig(os.path.join(experiments_dir, f"{filename_prefix}_learning_curves.png"))
    plt.show()
    plt.close()

def plot_confusion_matrix(cm, classes, config_details, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, base_filename='confusion_matrix'):
    plt.figure(figsize=(10, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Dynamically construct the filename from the configuration details
    config_string = "_".join([f"{k}_{v}" for k, v in config_details.items()])
    filename = f"{base_filename}_{config_string}.png"

    # Ensure the "experiments" directory exists and save the figure there
    experiments_dir = "experiments"
    os.makedirs(experiments_dir, exist_ok=True)
    plt.savefig(os.path.join(experiments_dir, filename), bbox_inches='tight')
    plt.close()

def log_metrics(train_loss, train_accuracy, val_loss, val_accuracy, early_stopping_epoch, model_training_time, config, filename="training_log.csv"):
    # Adjust the filename to include the "experiments" directory
    filename = os.path.join(experiments_dir, filename)
    
    # Check if the file already exists and has content (to decide whether to write the header)
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
    
    with open(filename, mode='a', newline='') as csv_file:
        fieldnames = ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'early_stopping_epoch', 'model_training_time', 'config']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'early_stopping_epoch': early_stopping_epoch,
            'model_training_time': model_training_time,
            'config': str(config)
        })

def config_to_string(config):
    config_items = ["{}_{}".format(key, value) for key, value in config.items()]
    return "_".join(config_items)

def main():
    # Transformations and DataLoader setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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
    if config['optims']== 'SGD':
        optimizer= optim.SGD(model.parameters(), lr= config['learning_rate'])
    elif config['optims']== 'Adam':
        optimizer= optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optims']== 'Adadelta':
        optimizer= optim.Adadelta(model.parameters(), lr=config['learning_rate'])
    elif config['optims']== 'Adagrad':
        optimizer= optim.Adagrad(model.parameters(), lr= config['learning_rate'])
    else:
        print("not valid optimizer")

    early_stopping = EarlyStopping(patience=config["patience"], verbose=True)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    start_time = time.time()
    early_stopping_epoch= None
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for _, (data, target) in enumerate(train_loader):
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


    # Assuming evaluate_model, model, test_loader, criterion are defined
    _, test_accuracy, all_preds, all_targets = evaluate_model(model, test_loader, criterion, device)
    cm = confusion_matrix(all_targets, all_preds)
    classes = list(range(10))

    # Correct usage of the plotting function
    filename_prefix = "confusion_matrix"  # Example prefix, adjust as needed
    plot_confusion_matrix(cm, classes, config, normalize=True, title='Normalized Confusion Matrix')

    filename_prefix = f"model_performance_{config_to_string(config)}"
    save_learning_curves(train_accuracies, val_accuracies, train_losses, val_losses, filename_prefix)


if __name__ == "__main__":
    # Ensure the "experiments" directory exists
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    main()
