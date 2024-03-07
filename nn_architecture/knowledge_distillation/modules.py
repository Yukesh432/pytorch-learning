import random
import numpy as np
from tqdm import tqdm
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# generator, decorator, OOP,, x, y transpose,,,  remained to be checked 

def initialize_network(n_x, n_h, n_y):
    np.random.seed(2)
    w1= np.random.randn(n_h, n_x) * 0.01
    b1= np.zeros((n_h, 1))
    w2= np.random.randn(n_y, n_h)* 0.01
    b2= np.zeros((n_y, 1))

    parameters= {"W1": w1, "b1": b1, "W2": w2, "b2": b2}
    return parameters
# def initialize_network(n_x, n_h, n_y):
#     np.random.seed(2)
    
#     w1 = np.random.randn(n_h, n_x) * np.sqrt(2. / n_x)  # He initialization for the first layer
#     b1 = np.zeros((n_h, 1))
    
#     w2 = np.random.randn(n_y, n_h) * np.sqrt(2. / n_h)  # He initialization for the second layer
#     b2 = np.zeros((n_y, 1))

#     parameters = {"W1": w1, "b1": b1, "W2": w2, "b2": b2}
#     return parameters


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
    one_hot_encoded[Y, np.arange(Y.size)] = 1
    return one_hot_encoded

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Improve numerical stability
    return e_z / e_z.sum(axis=0, keepdims=True)

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = softmax(z2)

    cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2, cache

def compute_loss(A2, Y):
    m = Y.shape[1]
    epsilon = 1e-6  # Small constant to avoid log(0)
    # Using the cross-entropy loss for multi-class classification
    logprobs = -np.log(A2 + epsilon) * Y
    loss = np.sum(logprobs) / m
    return loss


def optimizer(parameters, cache, X, Y, learning_rate=0.001, epsilon=1e-8):
    """
    This function performs the optimization step using the AdaGrad optimizer.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    cache -- a dictionary containing "z1", "a1", "z2" and "a2".
    X -- input data of shape (number of features, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    learning_rate -- learning rate of the gradient descent update rule
    epsilon -- small constant to avoid division by zero in AdaGrad update
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    grads -- python dictionary containing your gradients 
    """
    m = X.shape[1]
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Retrieve cache for gradients
    A1 = cache['a1']
    A2 = cache['a2']

    # Backpropagation: Calculate dW1, db1, dW2, db2
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (A1 > 0)  # ReLU derivative part
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # Initialize the AdaGrad accumulators if they don't exist
    if "accum_W1" not in parameters:
        parameters["accum_W1"] = np.zeros(W1.shape)
        parameters["accum_b1"] = np.zeros(b1.shape)
        parameters["accum_W2"] = np.zeros(W2.shape)
        parameters["accum_b2"] = np.zeros(b2.shape)

    # AdaGrad update
    parameters["accum_W1"] += dW1**2
    parameters["accum_b1"] += db1**2
    parameters["accum_W2"] += dW2**2
    parameters["accum_b2"] += db2**2

    W1 -= learning_rate * dW1 / (np.sqrt(parameters["accum_W1"]) + epsilon)
    b1 -= learning_rate * db1 / (np.sqrt(parameters["accum_b1"]) + epsilon)
    W2 -= learning_rate * dW2 / (np.sqrt(parameters["accum_W2"]) + epsilon)
    b2 -= learning_rate * db2 / (np.sqrt(parameters["accum_b2"]) + epsilon)

    # Update parameters
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "accum_W1": parameters["accum_W1"], "accum_b1": parameters["accum_b1"], "accum_W2": parameters["accum_W2"], "accum_b2": parameters["accum_b2"]}

    # Store grads in a dictionary
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return parameters, grads


# Now in your nn_model function, replace the backward_propagation and update_parameters calls with the optimizer function

def nn_model(X_train, Y_train, X_test, Y_test, n_h, num_iterations=1000, print_cost=False):
    print("Training started...")
    np.random.seed(3)
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]  # This should be the number of classes, ensure Y_train is one-hot encoded

    parameters = initialize_network(n_x, n_h, n_y)
    
    train_losses = []
    test_losses = []
    accuracies = []
    
    for i in tqdm(range(num_iterations)):
        # Forward propagation
        A2, cache = forward_propagation(X_train, parameters)
        cost_train = compute_loss(A2, Y_train)
        
        # Backward propagation and parameters update using the optimizer function
        parameters, grads = optimizer(parameters, cache, X_train, Y_train, learning_rate=0.01)
    
        # Validation loss
        A2_test, _ = forward_propagation(X_test, parameters)
        cost_test = compute_loss(A2_test, Y_test)
        
        if print_cost and i % 100 == 0:
            predictions_test = predict(parameters, X_test)
            # Calculate the accuracy using the one-hot encoded labels and the predictions
            accuracy = accuracy_score(Y_test.argmax(axis=0), predictions_test)
            accuracies.append(accuracy)
            train_losses.append(cost_train)
            test_losses.append(cost_test)
            print(f"Iteration {i} | Train Loss: {cost_train:.4f} | Test Loss: {cost_test:.4f} | Accuracy: {accuracy * 100:.2f}%")
        
    cm = confusion_matrix(Y_val.argmax(axis=0), predictions_test)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    return parameters, train_losses, test_losses, accuracies

def predict(parameters, X):
    # Use forward_propagation to get A2
    A2, cache = forward_propagation(X, parameters)
    # Convert probabilities to class predictions
    predictions = np.argmax(A2, axis=0)
    return predictions

def get_data_from_loader(dataloader):
    images_list = []
    labels_list = []
    
    for images, labels in dataloader:
        # Flatten the images and convert to NumPy arrays
        images = images.view(images.size(0), -1).numpy()
        # Convert labels to NumPy arrays
        labels = labels.numpy()
        # Append to lists
        images_list.append(images)
        labels_list.append(labels)
    
    # Concatenate all the data
    images_np = np.concatenate(images_list, axis=0).T  # Transpose to match your nn_model's expected input
    labels_np = np.concatenate(labels_list, axis=0).reshape(1, -1)  # Reshape to a row vector for nn_model

    print("The images shape isss::::::::", images_np.shape)
    
    return images_np, labels_np



# Assuming the definition of get_data_from_loader, one_hot_encode, nn_model and other necessary functions are already available

if __name__ == "__main__":
    # Transformations for the image data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training and test datasets
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Data loaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64, shuffle=False)

    start_time = time.time()
    # Get training data in NumPy format
    X_train_np, y_train_np = get_data_from_loader(trainloader)

    # Train and validation split (using a subset of the training data as validation data for simplicity)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_np.T, y_train_np.T, test_size=0.2, random_state=42)

    # Prepare data for the model (Transpose to match the expected shape by the model)
    X_train = X_train.T
    X_val = X_val.T
    Y_train = Y_train.T
    Y_val = Y_val.T

    print(X_train.shape)
    print(X_val.shape)
    print(Y_train.shape)
    print(Y_val.shape)
    print("Finished shape...........")

    # Assuming Y_train and Y_test are numpy arrays containing the labels for the training and test sets
    C = 10  # Number of classes

    Y_train_one_hot = one_hot_encode(Y_train, C)
    Y_test_one_hot = one_hot_encode(Y_val, C)

    parameters, train_losses, test_losses, accuracies = nn_model(X_train, Y_train_one_hot, X_val, Y_test_one_hot, n_h=128, num_iterations=500, print_cost=True)
    # Plotting

    plt.figure(figsize=(12, 6))

    # Epoch vs Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(accuracies)
    plt.xlabel('Iterations (per hundreds)')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs. Accuracy')

    # Epoch vs Loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Iterations (per hundreds)')
    plt.ylabel('Loss')
    plt.title('Epoch vs. Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    end_time = time.time()
    print(f"The time taken by model to train:: {end_time - start_time}")

# vector/matrix:: Representation 