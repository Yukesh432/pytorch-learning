import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid

from models import NaturalSceneClassification
from utils import fit, evaluate, accuracy

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break

def main():
    # Data directory
    data_dir = "../data/intel-image-classification/seg_train/seg_train/"
    test_data_dir = "../data/intel-image-classification/seg_test/seg_test"

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    # Loading datasets
    dataset = ImageFolder(data_dir, transform=transform)
    test_dataset = ImageFolder(test_data_dir, transform=transform)

    # Splitting dataset
    batch_size = 128
    val_size = 2000
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Length of Train Data: {len(train_data)}")
    print(f"Length of Validation Data: {len(val_data)}")

    # Data loaders
    train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)

    # show_batch(train_dl)

    # Model, optimizer, and training
    model = NaturalSceneClassification()
    
    epochs = 2  # example value, set as needed
    lr = 0.01   # example value, set as needed
    history = fit(epochs, lr, model, train_dl, val_dl)

    # Save the model
    torch.save(model.state_dict(), 'natural_scene_classification.pth')

if __name__ == '__main__':
    main()
