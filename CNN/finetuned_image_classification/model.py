import torchvision.models as models
import torch.nn as nn

def get_model(num_classes, pretrained=True):
    # Load a pre-trained model
    model = models.resnet50(pretrained=pretrained)

    # Replace the classifier
    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Linear(num_features, num_classes)
    return model

# print(get_model(10))