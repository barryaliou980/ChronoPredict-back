"""Module de prédiction par image pour les maladies chroniques.

Charge les modèles CNN entraînés et effectue des prédictions
à partir d'images médicales (rétinopathie, radiographie, lésions cutanées).
"""

import json
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# Répertoire des modèles sauvegardés
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Statistiques ImageNet pour la normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# Cache des modèles chargés en mémoire (singleton)
_loaded_models: dict[str, tuple[nn.Module, list[str]]] = {}

# Correspondance entre le type de modèle et le nom de fichier
MODEL_FILES = {
    "retinopathy": "retinopathy_model",
    "chest_xray": "chest_xray_model",
    "skin_lesion": "skin_lesion_model",
}

# Descriptions des modèles pour les réponses API
MODEL_DESCRIPTIONS = {
    "retinopathy": "Analyse de rétinopathie diabétique à partir d'images du fond d'oeil",
    "chest_xray": "Détection de pneumonie à partir de radiographies thoraciques",
    "skin_lesion": "Classification de lésions cutanées à partir d'images dermoscopiques",
}

# Correspondance maladie -> type de modèle d'image
DISEASE_TO_IMAGE_MODEL = {
    "Diabetes": "retinopathy",
    "Pneumonia": "chest_xray",
    "Psoriasis": "skin_lesion",
    "Acne": "skin_lesion",
    "Impetigo": "skin_lesion",
    "Fungal infection": "skin_lesion",
}


def get_inference_transform() -> transforms.Compose:
    """Retourne le pipeline de transformations pour l'inférence.

    Identique aux transformations de validation (sans augmentation).
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _create_model(num_classes: int) -> nn.Module:
    """Crée l'architecture ResNet18 avec la couche finale adaptée.

    Args:
        num_classes: Nombre de classes de sortie.

    Returns:
        Modèle ResNet18 (sans poids pré-entraînés, les poids seront chargés).
    """
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes),
    )
    return model


def load_model(model_type: str) -> tuple[nn.Module, list[str]]:
    """Charge un modèle entraîné depuis le disque (avec mise en cache).

    Args:
        model_type: Type de modèle ("retinopathy", "chest_xray", "skin_lesion").

    Returns:
        Tuple (modèle, liste des noms de classes).

    Raises:
        FileNotFoundError: Si le modèle n'a pas été entraîné.
        ValueError: Si le type de modèle est invalide.
    """
    global _loaded_models

    # Vérifier le cache
    if model_type in _loaded_models:
        return _loaded_models[model_type]

    # Valider le type de modèle
    if model_type not in MODEL_FILES:
        raise ValueError(
            f"Type de modèle invalide: '{model_type}'. "
            f"Types disponibles: {list(MODEL_FILES.keys())}"
        )

    model_filename = MODEL_FILES[model_type]
    model_path = MODELS_DIR / f"{model_filename}.pth"
    classes_path = MODELS_DIR / f"{model_filename}_classes.json"

    # Vérifier que les fichiers existent
    if not model_path.exists():
        raise FileNotFoundError(
            f"Le modèle '{model_type}' n'a pas été entraîné. "
            f"Fichier attendu: {model_path}. "
            f"Exécutez 'python -m app.ml.train_image' pour entraîner les modèles."
        )

    if not classes_path.exists():
        raise FileNotFoundError(
            f"Le fichier de classes pour '{model_type}' est introuvable: {classes_path}"
        )

    # Charger les noms de classes
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Créer et charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Mettre en cache
    _loaded_models[model_type] = (model, class_names)

    return model, class_names


def is_model_available(model_type: str) -> bool:
    """Vérifie si un modèle entraîné est disponible sur le disque.

    Args:
        model_type: Type de modèle à vérifier.

    Returns:
        True si le fichier .pth et le fichier de classes existent.
    """
    if model_type not in MODEL_FILES:
        return False

    model_filename = MODEL_FILES[model_type]
    model_path = MODELS_DIR / f"{model_filename}.pth"
    classes_path = MODELS_DIR / f"{model_filename}_classes.json"

    return model_path.exists() and classes_path.exists()


def get_model_classes(model_type: str) -> list[str]:
    """Retourne les noms de classes d'un modèle (depuis le fichier JSON).

    Args:
        model_type: Type de modèle.

    Returns:
        Liste des noms de classes, ou liste vide si le modèle n'existe pas.
    """
    if model_type not in MODEL_FILES:
        return []

    classes_path = MODELS_DIR / f"{MODEL_FILES[model_type]}_classes.json"
    if not classes_path.exists():
        return []

    with open(classes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_from_image(image_bytes: bytes, model_type: str) -> dict:
    """Effectue une prédiction à partir des octets d'une image.

    Args:
        image_bytes: Contenu binaire de l'image.
        model_type: Type de modèle à utiliser.

    Returns:
        Dictionnaire avec les clés:
        - prediction: Classe prédite (str)
        - confidence: Confiance de la prédiction (float)
        - probabilities: Probabilités pour chaque classe (dict[str, float])
    """
    # Charger le modèle
    model, class_names = load_model(model_type)
    device = next(model.parameters()).device

    # Préparer l'image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = get_inference_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Effectuer la prédiction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    # Extraire les résultats
    confidence, predicted_idx = torch.max(probabilities, 0)

    # Construire le dictionnaire de probabilités
    prob_dict = {
        class_names[i]: round(probabilities[i].item(), 4)
        for i in range(len(class_names))
    }

    # Trier par probabilité décroissante
    prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

    return {
        "prediction": class_names[predicted_idx.item()],
        "confidence": round(confidence.item(), 4),
        "probabilities": prob_dict,
    }
