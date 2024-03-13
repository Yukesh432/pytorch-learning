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
            'batch_size': batch_size  # This should be updated based on your DataLoader configuration
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

    def update_parameters(self, grads):
        """
        Updates the parameters of the network using gradient descent.

        Parameters
        ----------
        grads : dict
            A dictionary containing the gradients of the loss function with respect to each parameter.
        """
        # Update each parameter in the direction of the negative gradient
        # This is done to minimize the loss function
        self.parameters['W1'] -= self.learning_rate * grads['dW1']  # Update weights of the first layer
        self.parameters['b1'] -= self.learning_rate * grads['db1']  # Update biases of the first layer
        self.parameters['W2'] -= self.learning_rate * grads['dW2']  # Update weights of the second (output) layer
        self.parameters['b2'] -= self.learning_rate * grads['db2']  # Update biases of the second (output) layer


    def train(self, trainloader, testloader, epochs=1000, print_cost=False, patience=10):
        """
        Trains the neural network using the provided training and test data loaders.

        Parameters
        ----------
        trainloader : DataLoader
            The DataLoader containing the training data.
        testloader : DataLoader
            The DataLoader containing the test data.
        n_h : int
            Number of neurons in the hidden layer.
        num_iterations : int, optional
            Number of epochs to train the network for (default is 1000).
        print_cost : bool, optional
            Whether to print the cost/loss after each epoch (default is False).
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped (default is 10).

        Returns
        -------
        tuple
            A tuple containing training losses, test losses, accuracies, and the epoch at which training was stopped.
        """
        # Initialize lists to track the loss and accuracy
        train_losses, test_losses, accuracies = [], [], []
        best_test_loss = np.inf  # Initialize best_test_loss to infinity
        early_stop_epoch = epochs  # Initialize early_stop_epoch to the max number of iterations
        no_improve_counter = 0  # Counter to track the number of epochs without improvement

        # Loop over the dataset multiple times
        for epoch in tqdm(range(epochs)):
            train_loss_accum = 0  # Accumulator for the training loss
            correct_preds_epoch = 0  # Counter for correct predictions
            total_preds_epoch = 0  # Counter for total predictions

            # Iterate over the training data
            for X_batch, Y_batch in trainloader:
                # Flatten the batch and transpose it to match our weight matrix dimensions
                X_batch_np = X_batch.view(X_batch.size(0), -1).numpy().T
                # Convert labels to one-hot encoding
                Y_batch_ohe = self.one_hot_encode(Y_batch.numpy(), self.n_y)

                # Perform forward propagation
                A2, cache = self.forward_propagation(X_batch_np)
                # Compute the loss
                cost_train = self.compute_loss(A2, Y_batch_ohe)
                # Accumulate the loss over the batch
                train_loss_accum += cost_train * X_batch.size(0)

                # Perform backpropagation to get gradients
                grads = self.backpropagation(cache, X_batch_np, Y_batch_ohe)
                # Update parameters using gradients
                self.update_parameters(grads)

                # Make predictions and count correct ones
                predictions = np.argmax(A2, axis=0)

                correct_preds = np.sum(predictions == np.argmax(Y_batch_ohe, axis=1))
                # accumulates correct predictions across all batches
                correct_preds_epoch += correct_preds
                # keeps track of the total number of predictions
                total_preds_epoch += Y_batch.size(0)

            # Calculate average training loss and accuracy for the epoch
            train_loss_avg = train_loss_accum / total_preds_epoch
            train_accuracy = correct_preds_epoch / total_preds_epoch * 100

            # Append losses and accuracy to lists
            train_losses.append(train_loss_avg)
            test_loss, test_accuracy,_, _ = self.evaluate(testloader)
            test_losses.append(test_loss)
            accuracies.append(test_accuracy)

            # Check for improvement and implement early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                early_stop_epoch = epoch  # Record the epoch number at stopping
                break

            # Optionally print the cost/loss
            if print_cost:
                print(f"Epoch {epoch} | Train Loss: {train_loss_avg:.4f} | Train Accuracy: {train_accuracy:.2f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

        # Update model_details with the stopping epoch
        self.model_details['stopping_epoch'] = early_stop_epoch

        # Return the loss and accuracy records, along with the stopping epoch
        return train_losses, test_losses, accuracies, early_stop_epoch


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
    
# Directory setup
model_save_dir = 'experiment_results/models'
evaluation_save_dir = 'experiment_results/evaluation'
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(evaluation_save_dir, exist_ok=True)

if __name__ == "__main__":
    if __name__ == "__main__":
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

    # Model configuration
    n_x = 28*28
    n_h = 64
    n_y = 10
    learning_rate = 0.01
    activation_function = 'relu'
    initialization_method = 'xavier'
    epochs = 10
    optimizer = 'sgd'  # Placeholder, adjust as per your model
    batch_size = 64  # Match DataLoader batch size

    model_details = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'initialization_method': initialization_method,
        'n_h': n_h,
        'optimizer': optimizer,
        'batch_size': batch_size
    }

    # Initialize neural network model
    nn_model = NeuralNetwork(n_x=n_x, n_h=n_h, n_y=n_y, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs,
                             optimizer= optimizer, activation_function=activation_function, initialization_method=initialization_method)

    # Train the model
    start_time = time.time()
    train_losses, test_losses, accuracies, stopping_epoch= nn_model.train(trainloader, testloader,
                                                            epochs=epochs, print_cost=True)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")
    
    # Update model details with the actual stopping epoch
    nn_model.model_details['stopping_epoch'] = stopping_epoch

    # Save the model parameters
    model_save_dir = 'experiment_results/models'
    nn_model.save_model(model_save_dir)

    # Evaluate the model after training
    test_loss, test_accuracy, y_true, y_pred = nn_model.evaluate(testloader)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%")

    nn_model.save_confusion_matrix(y_true, y_pred, model_details)
 
    # Plot training and validation loss and save
    evaluation_save_dir = 'experiment_results/evaluation'
    os.makedirs(evaluation_save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Validation accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    evaluation_filename = f"{learning_rate}_{stopping_epoch}_{initialization_method}_{activation_function}_{n_h}_{optimizer}_{batch_size}_evaluation.png"
    evaluation_filepath = os.path.join(evaluation_save_dir, evaluation_filename)
    plt.savefig(evaluation_filepath)
    plt.show()