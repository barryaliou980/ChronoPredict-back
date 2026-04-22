import torch
import torch.nn as nn
from torchvision import models

def creer_modele():
    # 1. On charge le modèle ResNet18 pré-entraîné
    modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. On fige les "neurones" existants pour conserver leurs connaissances de base
    for parametre in modele.parameters():
        parametre.requires_grad = False
        
    # 3. On modifie la toute dernière couche du réseau (la couche de classification)
    nombre_caracteristiques = modele.fc.in_features
    modele.fc = nn.Linear(nombre_caracteristiques, 2)
    
    return modele

if __name__ == "__main__":
    mon_ia = creer_modele()
    print("✅ Architecture du modèle chargée !")
    print(f"Aperçu de la dernière couche : {mon_ia.fc}")