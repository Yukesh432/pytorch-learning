import torch
from torchvision import transforms
from PIL import Image
from models import NaturalSceneClassification

def load_model(model_path):
    model = NaturalSceneClassification()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),  # Match the training size
        transforms.ToTensor(),          # Convert to Tensor
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_path, class_names):
    image = preprocess_image(image_path)
    with torch.no_grad():  # No need to track gradients
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

class_names = {
    0: 'building',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}
# Usage
model_path = 'natural_scene_classification.pth'
image_path = '/OIP.jfif'  
model = load_model(model_path)
prediction = predict(model, image_path, class_names)

print(f'Predicted class: {prediction}')
