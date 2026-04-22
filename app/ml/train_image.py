"""Script d'entraînement des modèles CNN pour la prédiction par image.

Utilise ResNet18 pré-entraîné (ImageNet) avec fine-tuning pour trois tâches :
- Rétinopathie diabétique (APTOS 2019) : 5 classes de sévérité
- Radiographie thoracique (Chest X-Ray) : Normal / Pneumonie
- Lésions cutanées (HAM10000) : 7 types de lésions
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Répertoires de base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Hyperparamètres par défaut
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

# Statistiques ImageNet pour la normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """Retourne les transformations d'image pour l'entraînement ou l'évaluation.

    Args:
        is_training: Si True, applique des augmentations (flip, rotation, etc.)

    Returns:
        Pipeline de transformations torchvision.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def create_resnet18(num_classes: int) -> nn.Module:
    """Crée un modèle ResNet18 pré-entraîné avec une couche finale adaptée.

    Args:
        num_classes: Nombre de classes de sortie.

    Returns:
        Modèle ResNet18 modifié.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Geler les couches convolutives (sauf la dernière couche)
    for param in model.parameters():
        param.requires_grad = False

    # Remplacer la couche fully-connected finale
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes),
    )

    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    model_name: str,
) -> nn.Module:
    """Entraîne un modèle CNN avec validation.

    Args:
        model: Le modèle PyTorch à entraîner.
        train_loader: DataLoader d'entraînement.
        val_loader: DataLoader de validation.
        num_epochs: Nombre d'époques.
        device: Device (cpu ou cuda).
        model_name: Nom du modèle pour l'affichage.

    Returns:
        Le modèle entraîné avec les meilleurs poids.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # --- Phase d'entraînement ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)

        train_loss = running_loss / total_samples
        train_acc = running_corrects / total_samples

        # --- Phase de validation ---
        model.eval()
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)

        val_acc = val_corrects / val_total

        print(
            f"  Époque {epoch + 1}/{num_epochs} - "
            f"Loss: {train_loss:.4f} - "
            f"Acc train: {train_acc:.4f} - "
            f"Acc val: {val_acc:.4f}"
        )

        # Sauvegarder le meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

        scheduler.step()

    # Charger les meilleurs poids
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"  Meilleure accuracy de validation: {best_acc:.4f}")
    return model


def save_model(model: nn.Module, class_names: list[str], model_filename: str) -> None:
    """Sauvegarde le modèle et les noms de classes.

    Args:
        model: Modèle entraîné.
        class_names: Liste des noms de classes.
        model_filename: Nom du fichier (sans extension).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modèle
    model_path = MODELS_DIR / f"{model_filename}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Modèle sauvegardé: {model_path}")

    # Sauvegarder les noms de classes
    classes_path = MODELS_DIR / f"{model_filename}_classes.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"  Classes sauvegardées: {classes_path}")


def train_retinopathy(num_epochs: int = NUM_EPOCHS) -> None:
    """Entraîne le modèle de rétinopathie diabétique (APTOS 2019).

    Dataset attendu dans : backend/data/aptos2019/train/
    Sous-dossiers : 0, 1, 2, 3, 4 (niveaux de sévérité)
    """
    print("\n" + "=" * 60)
    print("  Entraînement - Rétinopathie Diabétique (APTOS 2019)")
    print("=" * 60)

    data_path = DATA_DIR / "aptos2019" / "train"
    if not data_path.exists():
        print(f"  Erreur: {data_path} introuvable.")
        print("  Téléchargez le dataset APTOS 2019 depuis Kaggle.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Charger les données
    train_dataset = datasets.ImageFolder(data_path, transform=get_transforms(True))
    val_dataset = datasets.ImageFolder(data_path, transform=get_transforms(False))

    # Séparer train/val (80/20)
    total = len(train_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Échantillons: {total} (train: {train_size}, val: {val_size})")

    # Créer et entraîner le modèle
    model = create_resnet18(num_classes).to(device)
    model = train_model(model, train_loader, val_loader, num_epochs, device, "retinopathy")

    # Sauvegarder
    save_model(model, class_names, "retinopathy_model")
    print("  Entraînement rétinopathie terminé!\n")


def train_chest_xray(num_epochs: int = NUM_EPOCHS) -> None:
    """Entraîne le modèle de radiographie thoracique (Chest X-Ray Pneumonia).

    Dataset attendu dans : backend/data/chest_xray/train/
    Sous-dossiers : NORMAL, PNEUMONIA
    """
    print("\n" + "=" * 60)
    print("  Entraînement - Radiographie Thoracique (Pneumonie)")
    print("=" * 60)

    data_path = DATA_DIR / "chest_xray" / "train"
    if not data_path.exists():
        print(f"  Erreur: {data_path} introuvable.")
        print("  Téléchargez le dataset Chest X-Ray depuis Kaggle.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Charger les données
    train_dataset = datasets.ImageFolder(data_path, transform=get_transforms(True))
    val_dataset = datasets.ImageFolder(data_path, transform=get_transforms(False))

    # Séparer train/val (80/20)
    total = len(train_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Échantillons: {total} (train: {train_size}, val: {val_size})")

    # Créer et entraîner le modèle
    model = create_resnet18(num_classes).to(device)
    model = train_model(model, train_loader, val_loader, num_epochs, device, "chest_xray")

    # Sauvegarder
    save_model(model, class_names, "chest_xray_model")
    print("  Entraînement radiographie terminé!\n")


def train_skin_lesion(num_epochs: int = NUM_EPOCHS) -> None:
    """Entraîne le modèle de lésions cutanées (HAM10000).

    Dataset attendu dans : backend/data/ham10000/train/
    Sous-dossiers : akiec, bcc, bkl, df, mel, nv, vasc
    """
    print("\n" + "=" * 60)
    print("  Entraînement - Lésions Cutanées (HAM10000)")
    print("=" * 60)

    data_path = DATA_DIR / "ham10000" / "train"
    if not data_path.exists():
        print(f"  Erreur: {data_path} introuvable.")
        print("  Téléchargez le dataset HAM10000 depuis Kaggle.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Charger les données
    train_dataset = datasets.ImageFolder(data_path, transform=get_transforms(True))
    val_dataset = datasets.ImageFolder(data_path, transform=get_transforms(False))

    # Séparer train/val (80/20)
    total = len(train_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Échantillons: {total} (train: {train_size}, val: {val_size})")

    # Créer et entraîner le modèle
    model = create_resnet18(num_classes).to(device)
    model = train_model(model, train_loader, val_loader, num_epochs, device, "skin_lesion")

    # Sauvegarder
    save_model(model, class_names, "skin_lesion_model")
    print("  Entraînement lésions cutanées terminé!\n")


def train_all(num_epochs: int = NUM_EPOCHS) -> None:
    """Entraîne les trois modèles d'image séquentiellement."""
    print("=" * 60)
    print("  Entraînement de tous les modèles d'image")
    print("=" * 60)

    train_retinopathy(num_epochs)
    train_chest_xray(num_epochs)
    train_skin_lesion(num_epochs)

    print("\n" + "=" * 60)
    print("  Tous les modèles d'image ont été entraînés!")
    print("=" * 60)


if __name__ == "__main__":
    # Nombre d'époques passé en argument (optionnel)
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_EPOCHS
    train_all(epochs)
