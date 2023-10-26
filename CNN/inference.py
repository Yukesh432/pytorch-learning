import torch
import torchvision.transforms as transforms
from PIL import Image  # If working with image data

# Load your pre-trained model
model = torch.load('cifar10_net_ann.pth')  # Replace with the path to your model

# Set the model to evaluation mode
model.eval()

# Load and preprocess your own data
# For example, if you have an image file, load it and apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your image (replace with your own data loading logic)
image = Image.open('Figure_1.png')
input_data = transform(image).unsqueeze(0)  # Add a batch dimension

# Move the input data to the same device as the model (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_data = input_data.to(device)

# Make predictions
with torch.no_grad():  # Disable gradient computation during inference
    output = model(input_data)
    print(output)
# You can now work with the 'output' tensor, which contains the model's predictions
