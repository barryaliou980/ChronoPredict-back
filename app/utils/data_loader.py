import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

chemin_donnees = 'data/chest_xray/train' # Adapte le chemin selon l'extraction

def preparer_chargeur_donnees(chemin_dossier, batch_size=32):
    if not os.path.exists(chemin_dossier):
        raise FileNotFoundError(f"Le dossier {chemin_dossier} est introuvable.")
        
    dataset = datasets.ImageFolder(root=chemin_dossier, transform=transformations)
    chargeur = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, chargeur

if __name__ == "__main__":
    dataset_entrainement, chargeur_entrainement = preparer_chargeur_donnees(chemin_donnees)
    print(f"✅ Dataset chargé avec succès !")
    print(f"Nombre d'images trouvées : {len(dataset_entrainement)}")
    print(f"Classes détectées par PyTorch : {dataset_entrainement.classes}")