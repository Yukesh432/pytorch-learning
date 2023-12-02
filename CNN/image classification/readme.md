## **Natural Scene Classification Project**

**Project Overview**

This project focuses on the classification of natural scenes using deep learning. The goal is to accurately classify images into various categories such as mountains, streets, forests, buildings, seas, and glaciers.

**Model Architecture**

The core of the project is the NaturalSceneClassification model, a convolutional neural network (CNN) built using PyTorch. The model architecture is defined in models.py and includes multiple convolutional layers followed by fully connected layers.

**Training**

The model is trained using the script in main.py. Training involves several epochs over the dataset, using loss and accuracy metrics to monitor performance. The trained model is saved as *naturalclassification.pth.*

**Inference**

For making predictions with the trained model, the *inference.py* script is used. It loads the trained model and processes input images to classify them into one of the predefined categories.
