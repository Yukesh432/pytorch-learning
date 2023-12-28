import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model

def preprocess_data(test_dir):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader

def infer(model, test_loader, device):
    model = model.to(device)
    results = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            results.append(preds.item())

    return results

def main():
    model_path = 'C:/Users/AIXI/OneDrive/Desktop/projects/fire-detection/fire_detection_model.pth'
    test_dir = 'C:/Users/AIXI/OneDrive/Desktop/projects/fire-detection/testimage'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path)
    test_loader = preprocess_data(test_dir)
    predictions = infer(model, test_loader, device)

    for pred in predictions:
        print("Fire Detected" if pred == 1 else "No Fire Detected")

if __name__ == "__main__":
    main()
