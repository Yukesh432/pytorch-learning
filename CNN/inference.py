import torch
import torchvision.transforms as transforms
from PIL import Image  # If working with image data
import torch
import torch.nn as nn

# Define an ANN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Load your pre-trained model
# model = torch.load('cifar10_net_ann_full.pth')  # Replace with the path to your model
model = torch.load('cifar10_net_ann_full_cpu.pth')  # Replace with the path to your model

# Set the model to evaluation mode
model.eval()

# Load and preprocess your own data
# For example, if you have an image file, load it and apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your image (replace with your own data loading logic)
image = Image.open('ss.jpg').convert('RGB')
image = image.resize((32, 32))

input_data = transform(image).unsqueeze(0)  # Add a batch dimension

# Move the input data to the same device as the model (CPU or GPU)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device= torch.device("cpu")
input_data = input_data.to(device)

# Make predictions
with torch.no_grad():  # Disable gradient computation during inference
    output = model(input_data)
    print(output)

    # Assuming 'output' is your model's output tensor
    predicted_class = torch.argmax(output, dim=1).item()


# CIFAR-10 class labels
class_labels = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

predicted_class_label = class_labels[predicted_class]

print(f"The predicted class is: {predicted_class_label}")