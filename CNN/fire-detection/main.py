import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import train_model, initialize_model, evaluate_model, plot_confusion_matrix

def main():
    data_dir = '/path/Fire-Detection'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("devide is ...................", device)
    

    dataloaders, dataset_sizes = load_data(data_dir)
    model = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5, device=device)

    # After training, evaluate the model
    conf_matrix, class_report = evaluate_model(model, dataloaders, device)
    print("Classification Report:\n", class_report)
    plot_confusion_matrix(conf_matrix)

    # Save the model
    torch.save(trained_model.state_dict(), 'fire_detection_model2.pth')

if __name__ == "__main__":
    main()
