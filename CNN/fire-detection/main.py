import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import train_model, initialize_model

def main():
    data_dir = 'C:/Users/AIXI/OneDrive/Desktop/projects/fire-detection/Fire-Detection'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("devide is ...................", device)
    

    dataloaders, dataset_sizes = load_data(data_dir)
    model = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device=device)

    # Save the model
    torch.save(trained_model.state_dict(), 'fire_detection_model.pth')

if __name__ == "__main__":
    main()
