import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


'''
Reference: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
'''
training_data= datasets.FashionMNIST(
    root= "data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data= datasets.FashionMNIST(
    root= "data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader= DataLoader(training_data, batch_size=64)
test_dataloader= DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.modules):
    def __init__(self):
        super().__init__()
        self.flatten= nn.Flatten()
        self.linear_relu_stack= nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x= self.flatten(x)
        logits= self.linear_relu_stack(x)
        return logits
    

model= NeuralNetwork()

#these are hypperparameters , which are adjustable . The purpose of hyperparameter tuuning is to make optimized model which helps
#in faster convergence.
learning_rate = 1e-3
batch_size = 64
epochs = 5

#loss Function: It measures the difference in predicted value and actual value of the predicted answer
#Eg. Mean sqare error (nn.MSELoss), negative log likelihood(nn.NLLLoss)
# nn.CrossEntropyLoss combines nn.LogSoftmax and nn.LLLoss

# we pass our model output logits to the crossentropy loss which will noralize the logits and compute the prediction error
losss= nn.CrossEntropyLoss()

# now we initialize the optimizer
optimizer= torch.optim.SGD(model.parameters(), lr= learning_rate)


#we define the train_loop that loops over our optimization code 

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")