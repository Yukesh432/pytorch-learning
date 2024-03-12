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
    def __init__(self, n_x, n_h, n_y, learning_rate=0.01, activation_function='relu', initialization_method='random'):
        self.n_y = n_y  # Number of output units
        self.learning_rate = learning_rate
        self.activation_function_name = activation_function
        self.initialization_method = initialization_method
        self.parameters = self.initialize_network(n_x, n_h, n_y, initialization_method)
    
    def initialize_network(self, n_x, n_h, n_y, initialization_method):
        np.random.seed(2)
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
        if self.activation_function_name == 'relu':
            return np.maximum(0, z)
        elif self.activation_function_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_function_name == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError("Unsupported activation function")

    def softmax(self, z):
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
        W1, b1, W2, b2 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'], self.parameters['b2']
        Z1 = np.dot(W1, X) + b1
        A1 = self.activation_function(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.softmax(Z2)
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache
    
    def compute_loss(self, A2, Y):
        m = Y.shape[1]
        epsilon = 1e-6  # Small constant to avoid log(0)
        Y= Y.T
        logprobs = -np.log(A2 + epsilon) * Y
        loss = np.sum(logprobs) / m
        return loss

    def backpropagation(self, cache, X, Y):
        m = X.shape[1]
        W1, W2 = self.parameters['W1'], self.parameters['W2']
        A1, A2 = cache['A1'], cache['A2']
        Y= Y.T
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(W2.T, dZ2)
        if self.activation_function_name == 'relu':
            dZ1 = np.array(dA1, copy=True)
            dZ1[A1 <= 0] = 0
        elif self.activation_function_name in ['sigmoid', 'tanh']:
            dZ1 = dA1 * A1 * (1 - A1)  # For sigmoid and roughly for tanh

        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return grads

    def update_parameters(self, grads):
        self.parameters['W1'] -= self.learning_rate * grads['dW1']
        self.parameters['b1'] -= self.learning_rate * grads['db1']
        self.parameters['W2'] -= self.learning_rate * grads['dW2']
        self.parameters['b2'] -= self.learning_rate * grads['db2']

    def train(self, trainloader, testloader, n_h, num_iterations=1000, print_cost=False, patience=10):
        train_losses, test_losses, accuracies = [], [], []
        best_test_loss = np.inf
        no_improve_counter = 0

        for epoch in tqdm(range(num_iterations)):
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
                # correct_preds = np.sum(predictions == np.argmax(Y_batch_ohe, axis=0))
                correct_preds = np.sum(predictions == np.argmax(Y_batch_ohe, axis=1))
                correct_preds_epoch += correct_preds
                total_preds_epoch += Y_batch.size(0)

            train_loss_avg = train_loss_accum / total_preds_epoch
            train_accuracy = correct_preds_epoch / total_preds_epoch * 100
            train_losses.append(train_loss_avg)

            test_loss, test_accuracy = self.evaluate(testloader)
            test_losses.append(test_loss)
            accuracies.append(test_accuracy)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

            if print_cost:
                print(f"Epoch {epoch} | Train Loss: {train_loss_avg:.4f} | Train Accuracy: {train_accuracy:.2f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

        return train_losses, test_losses, accuracies

    def evaluate(self, dataloader, model_details= None):
        test_loss_accum = 0
        correct_preds = 0
        total_preds = 0
        y_true, y_pred = [], []

        for X_test, Y_test in dataloader:
            X_test_np = X_test.view(X_test.size(0), -1).numpy().T
            Y_test_ohe = self.one_hot_encode(Y_test.numpy(), self.n_y)
            A2_test, _ = self.forward_propagation(X_test_np)
            cost_test = self.compute_loss(A2_test, Y_test_ohe)
            test_loss_accum += cost_test * X_test.size(0)

            predictions_test = np.argmax(A2_test, axis=0)
            correct_preds += np.sum(predictions_test == Y_test.numpy())
            total_preds += Y_test.size(0)

            y_true.extend(Y_test.numpy())
            y_pred.extend(predictions_test)

        average_test_loss = test_loss_accum / total_preds
        test_accuracy = correct_preds / total_preds * 100

        # Generate and save confusion matrix
        self.save_confusion_matrix(y_true, y_pred, model_details)

        return average_test_loss, test_accuracy
    
    
    def save_confusion_matrix(self, y_true, y_pred, model_details):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        evaluation_save_dir = 'experiment_results/evaluation'
        os.makedirs(evaluation_save_dir, exist_ok=True)
        confusion_matrix_filename = f"{model_details['learning_rate']}_{model_details['num_iterations']}_{model_details['initialization_method']}_{model_details['n_h']}_{model_details['optimizer']}_{model_details['batch_size']}_confusion_matrix.png"
        confusion_matrix_filepath = os.path.join(evaluation_save_dir, confusion_matrix_filename)
        plt.savefig(confusion_matrix_filepath)
        plt.close()
    
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
    initialization_method = 'he'
    num_iterations = 10
    optimizer = 'sgd'  # Placeholder, adjust as per your model
    batch_size = 64  # Match DataLoader batch size

    model_details = {
        'learning_rate': learning_rate,
        'num_iterations': num_iterations,
        'initialization_method': initialization_method,
        'n_h': n_h,
        'optimizer': optimizer,
        'batch_size': batch_size
    }

    # Initialize neural network model
    nn_model = NeuralNetwork(n_x=n_x, n_h=n_h, n_y=n_y, learning_rate=learning_rate,
                             activation_function=activation_function, initialization_method=initialization_method)

    # Train the model
    start_time = time.time()
    train_losses, test_losses, accuracies = nn_model.train(trainloader, testloader,
                                                           n_h=n_h, num_iterations=num_iterations, print_cost=True)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")

    # Evaluate the model after training
    test_loss, test_accuracy = nn_model.evaluate(testloader, model_details)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%")

    # Save model parameters
    model_save_dir = 'experiment_results/models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_filename = f"{learning_rate}_{num_iterations}_{initialization_method}_{n_h}_{optimizer}_{batch_size}.npz"
    model_filepath = os.path.join(model_save_dir, model_filename)
    np.savez(model_filepath, **nn_model.parameters)

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

    evaluation_filename = f"{learning_rate}_{num_iterations}_{initialization_method}_{n_h}_{optimizer}_{batch_size}_evaluation.png"
    evaluation_filepath = os.path.join(evaluation_save_dir, evaluation_filename)
    plt.savefig(evaluation_filepath)
    plt.show()