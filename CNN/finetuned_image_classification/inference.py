import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import os

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_path):
    with torch.no_grad():
        image = load_image(image_path)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return CIFAR10_CLASSES[predicted.item()]

def predict_on_folder(model, folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add more image formats if needed
            image_path = os.path.join(folder_path, filename)
            predicted_class = predict(model, image_path)
            print(f'Image: {filename}, Predicted class: {predicted_class}')

# Load the model and set to evaluation mode
model = get_model(num_classes=10)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Folder containing the images
folder_path = './mydata'  # Replace with your folder path

# Perform inference on all images in the folder
predict_on_folder(model, folder_path)
