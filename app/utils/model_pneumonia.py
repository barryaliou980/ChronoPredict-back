import torch.nn as nn
from torchvision import models

def create_pneumonia_model():
    # Chargement de ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # On fige les couches de base
    for param in model.parameters():
        param.requires_grad = False
        
    # Couche de sortie : 2 classes (Normal / Pneumonie)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model