import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# Les transformations standardisées pour ResNet18
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def prepare_pneumonia_loaders(chemin_dossier, batch_size=32):
    """
    Charge les images, sépare 80% pour l'entraînement et 20% pour le test,
    et retourne les deux DataLoaders nécessaires pour les métriques.
    """
    if not os.path.exists(chemin_dossier):
        raise FileNotFoundError(f"❌ Le dossier {chemin_dossier} est introuvable.")
        
    # 1. Chargement de TOUTES les images du dossier
    dataset_complet = datasets.ImageFolder(root=chemin_dossier, transform=transformations)
    
    # 2. Calcul de la répartition (Split 80% / 20%)
    total_images = len(dataset_complet)
    train_size = int(0.8 * total_images)
    test_size = total_images - train_size
    
    print(f"📂 PNEUMONIE : {total_images} images trouvées.")
    print(f"   - 🏋️ Entraînement : {train_size} images")
    print(f"   - 🧪 Test/Validation : {test_size} images")
    
    # 3. Séparation aléatoire du dataset
    train_dataset, test_dataset = random_split(dataset_complet, [train_size, test_size])
    
    # 4. Création des chargeurs (Seul le train a besoin d'être mélangé)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Bloc de test rapide (s'exécute uniquement si tu lances ce fichier directement)
if __name__ == "__main__":
    chemin_donnees = 'data/chest_xray/train'
    try:
        train_loader, test_loader = prepare_pneumonia_loaders(chemin_donnees)
        print("✅ DataLoaders préparés avec succès !")
    except Exception as e:
        print(e)