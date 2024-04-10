import torchvision.models as models
import torch.nn as nn

def get_model(num_classes, pretrained=True):
    """
    Load a pre-trained ResNet-50 model and modify its classifier to match the number of classes.
    
    Parameters
    ----------
    num_classes : int
        The number of output classes for the model.
    pretrained : bool, optional
        A boolean value that if True, loads a model pre-trained on the ImageNet dataset. The default is True.
    
    Returns
    -------
    model : torch.nn.Module
        The modified ResNet-50 model adapted for the specified number of classes.
    
    Notes
    -----
    This function modifies the fully connected layer (fc) of the ResNet-50 model to make it suitable for 
    the given number of output classes. It is particularly useful when fine-tuning a pre-trained model 
    for a different classification task.
    """
    
    # Load a pre-trained ResNet-50 model. If pretrained=True, the model weights are loaded as trained on ImageNet.
    model = models.resnet50(pretrained=pretrained)

    # Extract the number of input features the original classifier was expecting.
    num_features = model.fc.in_features

    # Debugging print statement to see the number of input features to the classifier.
    # Useful to understand model architecture adjustments.
    print(num_features)

    # Replace the model's fully connected layer with a new one suited for num_classes output classes.
    # This adapts the model for a different number of classes than it was originally trained for.
    model.fc = nn.Linear(num_features, num_classes)

    return model

# Example usage: Print the modified model to inspect its architecture.
# print(get_model(10))
