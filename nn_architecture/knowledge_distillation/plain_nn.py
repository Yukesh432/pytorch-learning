# fix the learning rate, fix the epoch, change hidden-layer activations: - sigmoid, relu, tanh, PReLU:: 
# note down accuracy, precision, recall

# fix the learning rate, fix the epoch, choose the best hidden layer activation from above, ,, change the OPTIMIZER
# Observe learning curve vs epoch:::: best optimizer would converge fase(i.e in fewer epoch)
# change the learning rate::: same observation
# chage the initialization::: same obsv
import numpy as np


class NeuralNetwork:
    def __init__(self, n_x, n_h, n_y):

        np.random.seed(2)


nn1= NeuralNetwork()
parameters= nn1.initialize_network_param(initialize= 'he-init', input_feature= 720, hidden_node= 10, output_node=3)

for i in range(num_iterations):
    cache= nn1.forward_propagate(parameters)
    loss= nn1.compute_loss(loss= 'cross-entropy')
    nn1.backpropagate(optimizer= 'adam' , learning_rate= 0.01, parameters, cache)