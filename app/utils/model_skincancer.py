import torch.nn as nn
from torchvision import models

def create_skincancer_model():
    # Chargement de ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Note : Pour le cancer de la peau, on peut choisir de ne PAS figer 
    # toutes les couches si on veut plus de précision, mais restons simple pour l'instant.
    for param in model.parameters():
        param.requires_grad = False
        
    # Couche de sortie : 7 classes (les 7 types de lésions du CSV)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    return model