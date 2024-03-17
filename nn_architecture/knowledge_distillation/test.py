import numpy as np
import os
import csv
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns


class NeuralNetwork:
    """
    Represents a simple two-layer neural network.

    Attributes
    ----------
    n_y : int
        Number of output units.
    learning_rate : float
        Learning rate for gradient descent optimization.
    activation_function_name : str
        Name of the activation function to use ('relu', 'sigmoid', or 'tanh').
    initialization_method : str
        Method to use for initializing weights ('he', 'xavier', or 'random').
    parameters : dict
        Dictionary storing the initialized network parameters.

    Parameters
    ----------
    n_x : int
        Number of input features.
    n_h : int
        Number of hidden units.
    n_y : int
        Number of output units.
    learning_rate : float, optional
        Learning rate for optimization (default is 0.01).
    activation_function : str, optional
        Name of the activation function (default is 'relu').
    initialization_method : str, optional
        Method for weights initialization (default is 'random').

    """
    def __init__(self, n_x, n_h, n_y, batch_size=32, learning_rate=0.01, epochs=10, optimizer='sgd', activation_function='relu', initialization_method='random'):
        self.n_y = n_y  # Number of output units
        self.batch_size= batch_size
        self.learning_rate = learning_rate
        self.epochs= epochs
        self.optimizer= optimizer
        self.activation_function_name = activation_function
        self.initialization_method = initialization_method
        self.parameters = self.initialize_network(n_x, n_h, n_y, initialization_method)
        self.evaluation_save_dir = 'experiment_results/evaluation'
        # Initialize model details
        self.model_details = {
            'learning_rate': learning_rate,
            'epochs': epochs,  # This should be updated during training
            'initialization_method': initialization_method,
            'n_h': n_h,
            'optimizer': optimizer,  # This should be set if you have an optimizer
            'batch_size': batch_size,  # This should be updated based on your DataLoader configuration
            'activation_function': activation_function
        }

    def get_model_details(self):
        """
        Returns the model details.

        Returns
        -------
        dict
            A dictionary containing details about the model's configuration.
        """
        return self.model_details    

    
    def initialize_network(self, n_x, n_h, n_y, initialization_method):
        """
        Initializes the weights and biases of the network.

        Parameters
        ----------
        n_x : int
            Number of input features.
        n_h : int
            Number of hidden units.
        n_y : int
            Number of output units.
        initialization_method : str
            Method to use for initializing weights ('he', 'xavier', 'random').

        Returns
        -------
        dict
            A dictionary containing the initialized weights and biases ('W1', 'b1', 'W2', 'b2').

        """

        np.random.seed(20) # Set seed for reproducible results
        if initialization_method == 'he':
            w1 = np.random.randn(n_h, n_x) * np.sqrt(2. / n_x)
            w2 = np.random.randn(n_y, n_h) * np.sqrt(2. / n_h)
        elif initialization_method == 'xavier':
            w1 = np.random.randn(n_h, n_x) * np.sqrt(1. / n_x)
            w2 = np.random.randn(n_y, n_h) * np.sqrt(1. / n_h)
        else:  # default or 'random' initialization
            w1 = np.random.randn(n_h, n_x) * 0.01
            w2 = np.random.randn(n_y, n_h) * 0.01
        b1 = np.zeros((n_h, 1))
        b2 = np.zeros((n_y, 1))
        return {"W1": w1, "b1": b1, "W2": w2, "b2": b2}

    def activation_function(self, z):
        """
        Computes the activation function for a given input.

        The function to be applied is determined by the `activation_function_name` attribute.

        Parameters
        ----------
        z : np.ndarray
            The input matrix to which the activation function is applied.

        Returns
        -------
        np.ndarray
            The output matrix after applying the activation function.

        Raises
        ------
        ValueError
            If an unsupported activation function name is provided.
        """
        if self.activation_function_name == 'relu':
            # ReLU activation function, returns max(0, z) element-wise
            return np.maximum(0, z)
        elif self.activation_function_name == 'sigmoid':
            # Sigmoid activation function, returns 1 / (1 + exp(-z)) element-wise
            return 1 / (1 + np.exp(-z))
        elif self.activation_function_name == 'tanh':
            # Hyperbolic tangent activation function, returns tanh(z) element-wise
            return np.tanh(z)
        else:
            # Raises an error if the activation function is not supported
            raise ValueError("Unsupported activation function")

    def softmax(self, z):
        """
    Computes the softmax function for each column of the input matrix.

    The softmax function is a generalization of the logistic function that "squashes" 
    a K-dimensional vector z of arbitrary real values to a K-dimensional vector 
    sigma(z) of real values in the range (0, 1) that add up to 1.

    Parameters
    ----------
    z : np.ndarray
        The input matrix to which the softmax function is applied.

    Returns
    -------
    np.ndarray
        The output matrix after applying the softmax function.
    """
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / e_z.sum(axis=0, keepdims=True)
    
    @staticmethod
    def one_hot_encode(Y, C):
        """
        Perform one-hot encoding of a class label vector.

        Arguments:
        Y -- array-like, shape (number of examples,)
        C -- number of classes, an integer (MNIST has 10 classes)

        Returns:
        one_hot_encoded -- one-hot encoded matrix, shape (C, number of examples)
        """
        Y = np.array(Y, dtype=int)
        one_hot_encoded = np.zeros((C, Y.size))
        one_hot_encoded[Y,np.arange(Y.size)] = 1
        return one_hot_encoded.T  # Transpose to match expected shape

    def forward_propagation(self, X):
        """
        Performs forward propagation through the network.

        Parameters
        ----------
        X : ndarray
            The input data.

        Returns
        -------
        A2 : ndarray
            The output from the last layer after forward propagation.
        cache : dict
            A dictionary containing the Z and A values from each layer.
        """
        # Retrieve parameters
        W1, b1, W2, b2 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'], self.parameters['b2']
        # First layer operations
        Z1 = np.dot(W1, X) + b1
        A1 = self.activation_function(Z1)
        # Second layer operations (output layer)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.softmax(Z2)
        # Cache values for backpropagation
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    
    def compute_loss(self, A2, Y):
        """
        Computes the cross-entropy loss.

        Parameters
        ----------
        A2 : ndarray
            The output from the last layer of the network (after forward propagation).
        Y : ndarray
            The labels for the input data.

        Returns
        -------
        loss : float
            The cross-entropy loss.
        """
        m = Y.shape[1]  # Number of examples
        epsilon = 1e-6  # Small constant to avoid log(0)
        Y = Y.T  # Transpose Y to match the shape of A2
        logprobs = -np.log(A2 + epsilon) * Y
        loss = np.sum(logprobs) / m
        return loss


    def backpropagation(self, cache, X, Y):
        """
    Performs backpropagation to compute gradients of the loss function with respect to the parameters.

    Parameters
    ----------
    cache : dict
        A dictionary containing the cached values from forward propagation.
    X : ndarray
        The input data.
    Y : ndarray
        The labels for the input data.

    Returns
    -------
    grads : dict
        A dictionary containing the gradients of the loss function with respect to the parameters.
    """
        # Calculate the number of examples
        m = X.shape[1]

        # Retrieve weights from the parameters dictionary
        W1, W2 = self.parameters['W1'], self.parameters['W2']

        # Retrieve activation values from the cache
        A1, A2 = cache['A1'], cache['A2']

        # Transpose the labels matrix to match the output's shape
        Y = Y.T

        # Compute gradient of the loss with respect to Z2
        # This represents the difference between the predicted output (A2) and the actual labels (Y)
        dZ2 = A2 - Y

        # Calculate the gradient of the loss with respect to W2
        # It is the product of the gradient with respect to Z2 and the activation of the first layer (A1), scaled by 1/m
        dW2 = np.dot(dZ2, A1.T) / m

        # Compute the gradient of the loss with respect to b2
        # Sum the gradients with respect to Z2 vertically, and scale by 1/m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Calculate the gradient of the loss with respect to A1
        # It involves backpropagating through the second layer weights (W2)
        dA1 = np.dot(W2.T, dZ2)

        # If the activation function is ReLU
        if self.activation_function_name == 'relu':
            # Initialize dZ1 similar to dA1
            dZ1 = np.array(dA1, copy=True)
            # Apply the derivative of ReLU: dZ1 is 0 where A1 is 0 or less, unchanged otherwise
            dZ1[A1 <= 0] = 0
        elif self.activation_function_name in ['sigmoid', 'tanh']:
            # For sigmoid and tanh, the derivative in terms of A1 is different
            # Note: This implementation for tanh is a simplification and not exact
            dZ1 = dA1 * A1 * (1 - A1)

        # Compute the gradient of the loss with respect to W1
        # It is the product of the gradient with respect to Z1 and the input X, scaled by 1/m
        dW1 = np.dot(dZ1, X.T) / m

        # Compute the gradient of the loss with respect to b1
        # Sum the gradients with respect to Z1 vertically, and scale by 1/m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Package the gradients in a dictionary for updating the parameters
        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

        return grads

    # def update_parameters(self, grads):
    #     """
    #     Updates the parameters of the network using gradient descent.

    #     Parameters
    #     ----------
    #     grads : dict
    #         A dictionary containing the gradients of the loss function with respect to each parameter.
    #     """
    #     # Update each parameter in the direction of the negative gradient
    #     # This is done to minimize the loss function
    #     self.parameters['W1'] -= self.learning_rate * grads['dW1']  # Update weights of the first layer
    #     self.parameters['b1'] -= self.learning_rate * grads['db1']  # Update biases of the first layer
    #     self.parameters['W2'] -= self.learning_rate * grads['dW2']  # Update weights of the second (output) layer
    #     self.parameters['b2'] -= self.learning_rate * grads['db2']  # Update biases of the second (output) layer
    def update_parameters(self, grads):
        """
        Updates the parameters of the network using RMSprop optimizer.

        Parameters
        ----------
        grads : dict
            A dictionary containing the gradients of the loss function with respect to each parameter.
        """
        # Initialize RMSprop parameters
        beta = 0.9  # RMSprop hyperparameter
        epsilon = 1e-8  # Small constant to avoid division by zero

        # Initialize the RMSprop parameters for each parameter
        if not hasattr(self, 'RMSprop_cache'):
            self.RMSprop_cache = {'W1': np.zeros_like(grads['dW1']), 'b1': np.zeros_like(grads['db1']),
                                'W2': np.zeros_like(grads['dW2']), 'b2': np.zeros_like(grads['db2'])}

        # Update RMSprop cache
        self.RMSprop_cache['W1'] = beta * self.RMSprop_cache['W1'] + (1 - beta) * np.square(grads['dW1'])
        self.RMSprop_cache['b1'] = beta * self.RMSprop_cache['b1'] + (1 - beta) * np.square(grads['db1'])
        self.RMSprop_cache['W2'] = beta * self.RMSprop_cache['W2'] + (1 - beta) * np.square(grads['dW2'])
        self.RMSprop_cache['b2'] = beta * self.RMSprop_cache['b2'] + (1 - beta) * np.square(grads['db2'])

        # Update parameters
        self.parameters['W1'] -= self.learning_rate * grads['dW1'] / (np.sqrt(self.RMSprop_cache['W1']) + epsilon)
        self.parameters['b1'] -= self.learning_rate * grads['db1'] / (np.sqrt(self.RMSprop_cache['b1']) + epsilon)
        self.parameters['W2'] -= self.learning_rate * grads['dW2'] / (np.sqrt(self.RMSprop_cache['W2']) + epsilon)
        self.parameters['b2'] -= self.learning_rate * grads['db2'] / (np.sqrt(self.RMSprop_cache['b2']) + epsilon)


    def train(self, trainloader, testloader, epochs=1000, print_cost=False, patience=10):
        """
        Trains the neural network using the provided training and test data loaders.

        Parameters
        ----------
        trainloader : DataLoader
            The DataLoader containing the training data.
        testloader : DataLoader
            The DataLoader containing the test data.
        epochs : int, optional
            Number of epochs to train the network for (default is 1000).
        print_cost : bool, optional
            Whether to print the cost/loss after each epoch (default is False).
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped (default is 10).

        Returns
        -------
        tuple of lists
            A tuple containing lists of training losses, test losses, training accuracies, test accuracies, and the epoch at which training was stopped.
        """
        train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
        best_test_loss = np.inf
        early_stop_epoch = epochs
        no_improve_counter = 0

        for epoch in tqdm(range(epochs)):
            train_loss_accum = 0
            correct_preds_epoch = 0
            total_preds_epoch = 0

            for X_batch, Y_batch in trainloader:
                X_batch_np = X_batch.view(X_batch.size(0), -1).numpy().T
                Y_batch_ohe = self.one_hot_encode(Y_batch.numpy(), self.n_y)

                A2, cache = self.forward_propagation(X_batch_np)
                cost_train = self.compute_loss(A2, Y_batch_ohe)
                train_loss_accum += cost_train * X_batch.size(0)

                grads = self.backpropagation(cache, X_batch_np, Y_batch_ohe)
                self.update_parameters(grads)

                predictions = np.argmax(A2, axis=0)
                correct_preds = np.sum(predictions == np.argmax(Y_batch_ohe, axis=1))
                correct_preds_epoch += correct_preds
                total_preds_epoch += Y_batch.size(0)

            train_loss_avg = train_loss_accum / total_preds_epoch
            train_accuracy = correct_preds_epoch / total_preds_epoch * 100
            train_losses.append(train_loss_avg)
            train_accuracies.append(train_accuracy)

            test_loss, test_accuracy, _, _ = self.evaluate(testloader)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                early_stop_epoch = epoch
                break

            if print_cost:
                print(f"Epoch {epoch} | Train Loss: {train_loss_avg:.4f} | Train Accuracy: {train_accuracy:.2f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

        self.model_details['stopping_epoch'] = early_stop_epoch
        return train_losses, test_losses, train_accuracies, test_accuracies, early_stop_epoch

    def evaluate(self, dataloader):
        """
        Evaluates the neural network on a given dataset.

        Parameters
        ----------
        dataloader : DataLoader
            A DataLoader containing the dataset to evaluate the model on.
        model_details : dict
            A dictionary containing details of the model. Used for saving the confusion matrix.

        Returns
        -------
        average_test_loss : float
            The average loss of the model on the test dataset.
        test_accuracy : float
            The accuracy of the model on the test dataset.
        """
        # Initialize variables to accumulate test losses and track correct predictions
        test_loss_accum = 0
        correct_preds = 0
        total_preds = 0
        y_true, y_pred = [], []

        # Loop through batches in the test dataset
        for X_test, Y_test in dataloader:
            # Flatten the input and convert it to a NumPy array for processing
            X_test_np = X_test.view(X_test.size(0), -1).numpy().T
            
            # Convert the labels to one-hot encoding
            Y_test_ohe = self.one_hot_encode(Y_test.numpy(), self.n_y)
            
            # Perform forward propagation to get the network's predictions
            A2_test, _ = self.forward_propagation(X_test_np)
            
            # Calculate the loss for this batch and accumulate it
            cost_test = self.compute_loss(A2_test, Y_test_ohe)
            test_loss_accum += cost_test * X_test.size(0)
            
            # Determine the predicted classes for this batch
            predictions_test = np.argmax(A2_test, axis=0)
            
            # Update the count of correct predictions and total predictions
            correct_preds += np.sum(predictions_test == Y_test.numpy())
            total_preds += Y_test.size(0)
            
            # Extend the true and predicted labels lists for later analysis
            y_true.extend(Y_test.numpy())
            y_pred.extend(predictions_test)

        # Calculate the average test loss and accuracy across all batches
        average_test_loss = test_loss_accum / total_preds
        test_accuracy = correct_preds / total_preds * 100

        # # Generate and save confusion matrix using the true and predicted labels
        # self.save_confusion_matrix(y_true, y_pred, self.model_details)

        return average_test_loss, test_accuracy, y_true, y_pred

    
    
    def save_confusion_matrix(self, y_true, y_pred, model_details):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        
        os.makedirs(self.evaluation_save_dir, exist_ok=True)
        confusion_matrix_filename = f"{model_details['learning_rate']}_{model_details['epochs']}_{model_details['initialization_method']}_{model_details['activation_function']}_{model_details['n_h']}_{model_details['optimizer']}_{model_details['batch_size']}_confusion_matrix.png"
        confusion_matrix_filepath = os.path.join(evaluation_save_dir, confusion_matrix_filename)
        plt.savefig(confusion_matrix_filepath)
        plt.close()

    def save_model(self, save_dir, filename=None):
        """
        Saves the model parameters to a file.

        Parameters
        ----------
        save_dir : str
            The directory where the model parameters will be saved.
        filename : str, optional
            The name of the file to save the model parameters. If not provided,
            a filename is generated based on model details.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate a default filename if none is provided
        if filename is None:
            filename = f"model_{self.model_details['learning_rate']}_{self.model_details['stopping_epoch']}_{self.model_details['initialization_method']}_{self.model_details['activation_function']}_{self.model_details['n_h']}_{self.model_details['optimizer']}_{self.model_details['batch_size']}.npz"
            
        filepath = os.path.join(save_dir, filename)
        
        # Save the model parameters
        np.savez(filepath, **self.parameters)
        
        print(f"Model saved to {filepath}")

    def log_model_performance(self, csv_file_path, training_accuracy, test_accuracy):
        """
        Logs the model's configuration and performance metrics to a CSV file.

        Parameters
        ----------
        csv_file_path : str
            The file path for the CSV log file.
        training_accuracy : float
            The training accuracy of the model.
        test_accuracy : float
            The test accuracy of the model.
        """
        # Check if the CSV file already exists
        file_exists = os.path.isfile(csv_file_path)
        
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # If the file does not exist, write the header first
            if not file_exists:
                writer.writerow(['Learning Rate', 'Epochs', 'Initialization Method', 'Number of Hidden Units', 'Optimizer', 'Batch Size', 'Stopping Epoch', 'Training Accuracy', 'Test Accuracy'])
            
            # Write the model details and performance metrics
            writer.writerow([
                self.model_details['learning_rate'],
                self.model_details['epochs'],
                self.model_details['initialization_method'],
                self.model_details['n_h'],
                self.model_details['optimizer'],
                self.model_details['batch_size'],
                self.model_details.get('stopping_epoch', 'N/A'),  # Use 'N/A' if stopping_epoch isn't in model_details
                training_accuracy,
                test_accuracy
            ])
    
class AblationStudy:
    def __init__(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader

    def run_study(self, configurations):
        """
        Runs the ablation study over the set of provided configurations.

        Parameters
        ----------
        configurations : list of dicts
            A list where each dict contains specific configuration settings for a model.
        """
        for config in configurations:
            print(f"Starting for following configuration: \n{config}")
            print("*"*100)
            # Initialize the model with the current configuration
            nn_model = NeuralNetwork(n_x=config['n_x'], n_h=config['n_h'], n_y=config['n_y'],
                                     learning_rate=config['learning_rate'], epochs=config['epochs'],
                                     optimizer=config['optimizer'], batch_size=config['batch_size'],
                                     activation_function=config['activation_function'],
                                     initialization_method=config['initialization_method'])

            # Train the model
            train_losses, test_losses, train_accuracies, test_accuracies, early_stop_epoch = nn_model.train(self.trainloader, self.testloader, 
                                                                                   epochs=config['epochs'], print_cost=True)
            # Evaluate the model after training
            test_loss, test_accuracy, y_true, y_pred = nn_model.evaluate(self.testloader)
            # Log the model's performance
            final_training_accuracy = train_accuracies[-1]
            csv_file_path = 'model_performance_log.csv'
            nn_model.log_model_performance(csv_file_path, final_training_accuracy, test_accuracy)
            # Save the model parameters
            model_save_dir = 'experiment_results/models'
            nn_model.save_model(model_save_dir)
            # Save the confusion matrix
            nn_model.save_confusion_matrix(y_true, y_pred, config)
            # Plot and save training and validation loss
            self.save_evaluation_plots(train_losses, test_losses, test_accuracies, config)

    @staticmethod
    def save_evaluation_plots(train_losses, test_losses, accuracies, config):
        """
        Saves plots of the training/validation loss and validation accuracy.

        Parameters
        ----------
        train_losses : list
            List of training losses per epoch.
        test_losses : list
            List of validation/test losses per epoch.
        accuracies : list
            List of validation/test accuracies per epoch.
        config : dict
            The configuration dictionary for the current experiment.
        """
        # Construct a base filename from the configuration details
        base_filename = f"lr{config['learning_rate']}_epochs{config['epochs']}_init{config['initialization_method']}_nh{config['n_h']}_act{config['activation_function']}_bs{config['batch_size']}"
        
        evaluation_save_dir = 'experiment_results/evaluation'
        
        # Plot training and validation loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        
        # Construct the filepath and save the plots
        plots_filepath = os.path.join(evaluation_save_dir, f"{base_filename}_plots.png")
        plt.savefig(plots_filepath)
        plt.close()  # Close the plot to free resources
        
        print(f"Saved evaluation plots to {plots_filepath}")


# Directory setup
model_save_dir = 'experiment_results/models'
evaluation_save_dir = 'experiment_results/evaluation'
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(evaluation_save_dir, exist_ok=True)

if __name__ == "__main__":
    from itertools import product
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Prepare DataLoaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64, shuffle=False)
    # learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    # hidden_units_options = [32, 64, 128, 256, 512]
    # batch_sizes = [32, 64, 128]
    # activation_function= ['relu', 'sigmoid', 'tanh']
    # Ensure other parameters are set
    n_x = 28*28  # Input size for MNIST
    n_y = 10     # Number of output classes for MNIST
    n_h= 64
    lr= 0.01
    epochs = 50000
    optimizer = 'rmsprop'
    batch_size = 64
    activation_function = 'relu'
    initialization_method = 'he'

    # Generate configuration
    configurations = [
    {'n_x': n_x, 'n_h': n_h, 'n_y': n_y, 'learning_rate': lr, 'epochs': epochs,
     'optimizer': optimizer, 'batch_size': batch_size, 'activation_function': activation_function,
     'initialization_method': initialization_method}
]

# Reviewing a few generated configurations
for i, config in enumerate(configurations[:5], 1):  # Just show the first 5 for brevity
    print(f"Config {i}: LR={config['learning_rate']}, Hidden Units={config['n_h']}, Batch Size={config['batch_size']}")

    #     # Define configurations for the ablation study
    # configurations = [
    #     {'n_x': 28*28, 'n_h': 64, 'n_y': 10, 'learning_rate': 0.01, 'epochs': 5,
    #      'optimizer': 'sgd', 'batch_size': 64, 'activation_function': 'relu', 'initialization_method': 'he'},
    #     {'n_x': 28*28, 'n_h': 128, 'n_y': 10, 'learning_rate': 0.005, 'epochs': 5,
    #      'optimizer': 'sgd', 'batch_size': 64, 'activation_function': 'sigmoid', 'initialization_method': 'xavier'},
    #     {'n_x': 28*28, 'n_h': 128, 'n_y': 10, 'learning_rate': 0.5, 'epochs': 5,
    #      'optimizer': 'sgd', 'batch_size': 64, 'activation_function': 'sigmoid', 'initialization_method': 'xavier'},
    #     # Add more configurations as needed for the study
    # ]

    # Initialize AblationStudy
    study = AblationStudy(trainloader, testloader)

    # Ensure the model save directory exists
    model_save_dir = 'experiment_results/models'
    os.makedirs(model_save_dir, exist_ok=True)

    # Ensure the evaluation save directory exists
    evaluation_save_dir = 'experiment_results/evaluation'
    os.makedirs(evaluation_save_dir, exist_ok=True)

    # Run the ablation study
    start_time = time.time()
    study.run_study(configurations)
    end_time = time.time()

    print(f"Completed ablation study in {(end_time - start_time)/60:.2f} minutes")