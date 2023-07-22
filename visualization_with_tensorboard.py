import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Reference: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

#transforms
transform= transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5))]
)

#datasets
training_data= torchvision.datasets.FashionMNIST('./data', download=True, train= True, transform= transform)
testing_data= torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

#dataloaders
train_loader= DataLoader(training_data, batch_size= 4, shuffle= True)
test_loader= DataLoader(testing_data, batch_size=4, shuffle=False)

#Output labels for classes
classes= ('T-shirt/Top', 'Trouser', 'Pullover', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

#helper function to show an image
def matplotlib_image(img, one_channel= False):
    if one_channel:
        img= img.mean(dim=0)
    img= img / 2+ 0.5   #unnormalize.....why??
    npimg= img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap= "Greys")
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))


#defining model architecture

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #input channel: 1, output channel: 6, kernel size: 5*5
        self.conv1= nn.Conv2d(1, 6,5)
        self.pool= nn.MaxPool2d(2,2)
        self.conv2= nn.Conv2d(6, 16, 5)
        self.fc1= nn.Linear(16*4*4, 120)
        self.fc2= nn.Linear(120, 84)
        self.fc3= nn.Linear(84, 10)

    def forward(self,x):
        x= self.pool(F.relu(self.conv1(x)))
        x= self.pool(F.relu(self.conv2(x)))
        x= x.view(-1, 16*4*4)
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
        return x
    

net= Net()

#defining optimizer and criteron
criterion= nn.CrossEntropyLoss()
optimizer= optim.SGD(net.parameters(), lr= 0.001, momentum=0.9)

#tensorboard setup

from torch.utils.tensorboard import SummaryWriter

writer= SummaryWriter('runs/fashion_mnist_experiment_1')

#Writing to TensorBoard

#getting some random training images
dataiter= iter(train_loader)

images, labels= next(dataiter)

#creating a grid of images
img_grid= torchvision.utils.make_grid(images)

#showing images
matplotlib_image(img_grid, one_channel=True)

#write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)



writer.add_graph(net, images)
writer.close()

#tracking model training with Tensorboard

#helper function

def images_to_probs(net, images):
    '''Generate predictions and corresponding probabilites from a trained network and a list of images'''

    output= net(images)
    # convert the output  probabilites to predicted class
    _, preds_tensor= torch.max(output,1)
    preds= np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):

    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_image(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times

    for i, data in enumerate(train_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(train_loader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(train_loader) + i)
            running_loss = 0.0
print('Finished Training')