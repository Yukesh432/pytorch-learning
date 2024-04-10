import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import os

# Define a list of class names according to the CIFAR-10 dataset.
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_image(image_path):
    """
    Load and preprocess an image from the given path.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
    
    Returns
    -------
    torch.Tensor
        The preprocessed image as a tensor, ready for model inference.
    """
    # Define the transformations to preprocess the image.
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256 pixels on the shortest side.
        transforms.CenterCrop(224),  # Crop the center of the image to 224x224 pixels.
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image.
    ])
    image = Image.open(image_path)  # Open the image file.
    image = transform(image).unsqueeze(0)  # Apply the transformations and add a batch dimension.
    return image

def predict(model, image_path):
    """
    Predict the class of an image using the given model.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    image_path : str
        Path to the image file.
    
    Returns
    -------
    str
        The predicted class of the image.
    """
    with torch.no_grad():  # Disable gradient calculation for inference.
        image = load_image(image_path)  # Load and preprocess the image.
        outputs = model(image)  # Get the model's predictions.
        _, predicted = torch.max(outputs, 1)  # Find the class with the highest prediction score.
        return CIFAR10_CLASSES[predicted.item()]  # Return the name of the predicted class.

def predict_on_folder(model, folder_path):
    """
    Perform inference on all supported image files in the given folder.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    folder_path : str
        Path to the folder containing image files.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for supported image formats.
            image_path = os.path.join(folder_path, filename)  # Construct the full image path.
            predicted_class = predict(model, image_path)  # Predict the class of the image.
            print(f'Image: {filename}, Predicted class: {predicted_class}')  # Print the prediction.

# Load the model and set it to evaluation mode.
model = get_model(num_classes=10)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Specify the folder containing the images to be inferred.
folder_path = './mydata'  # Replace with the path to your folder.

# Perform inference on all images in the specified folder.
predict_on_folder(model, folder_path)
