import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_skincancer_loaders(train_path, batch_size=32):
    """
    Prépare les DataLoaders spécifiquement pour le dataset de cancer de la peau (HAM10000).
    """
    print("⚙️ Initialisation des DataLoaders pour le cancer de la peau...")

    # 1. Transformations pour l'ENTRAÎNEMENT (Avec augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20), # Tourne l'image jusqu'à 20 degrés
        transforms.ToTensor(),
        # Normalisation requise par ResNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Transformations pour le TEST (SANS augmentation)
    # L'IA doit être évaluée sur des images normales, sans effets
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Déduction du dossier de test
    base_dir = os.path.dirname(train_path)
    test_path = os.path.join(base_dir, 'test')

    # 3. Chargement avec ImageFolder
    # ImageFolder lit les dossiers (mel, nv, bkl...) et crée les 7 classes automatiquement
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    # 4. Création des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ DataLoader prêt : {len(train_dataset)} images détectées réparties en {len(train_dataset.classes)} classes.")
    print(f"📌 Classes trouvées : {train_dataset.classes}")
    
    return train_loader, test_loader