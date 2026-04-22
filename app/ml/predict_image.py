"""Module de prédiction par image pour les maladies chroniques.

Charge les modèles CNN entraînés et effectue des prédictions
à partir d'images médicales (radiographie thoracique, lésions cutanées).
"""

import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# Répertoires
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Constantes
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# Cache des modèles chargés
_loaded_models: dict[str, tuple[nn.Module, list[str]]] = {}

# Mapping des fichiers modèles existants
MODEL_FILES: dict[str, str] = {
    "chest_xray": "modele_pneumonie",
    "skin_lesion": "skin_cancer_model",
}

# Nombre de classes par modèle
MODEL_NUM_CLASSES: dict[str, int] = {
    "chest_xray": 2,
    "skin_lesion": 7,
}

# Noms de classes par modèle
MODEL_CLASSES: dict[str, list[str]] = {
    "chest_xray": ["NORMAL", "PNEUMONIA"],
    "skin_lesion": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
}

# Descriptions des modèles
MODEL_DESCRIPTIONS: dict[str, str] = {
    "chest_xray": "Détection de pneumonie à partir de radiographies thoraciques",
    "skin_lesion": "Classification de lésions cutanées à partir d'images dermoscopiques",
}

# Mapping maladie -> modèle image
DISEASE_TO_IMAGE_MODEL: dict[str, str] = {
    "Pneumonia": "chest_xray",
    "Psoriasis": "skin_lesion",
    "Acne": "skin_lesion",
    "Impetigo": "skin_lesion",
    "Fungal infection": "skin_lesion",
}


def get_inference_transform() -> transforms.Compose:
    """Retourne le pipeline de transformations pour l'inférence."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _create_model(num_classes: int) -> nn.Module:
    """Crée l'architecture ResNet18 avec la couche finale adaptée."""
    model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def load_model(model_type: str) -> tuple[nn.Module, list[str]]:
    """Charge un modèle entraîné depuis le disque (avec mise en cache).

    Args:
        model_type: Type de modèle ("chest_xray", "skin_lesion").

    Returns:
        Tuple (modèle, liste des noms de classes).
    """
    if model_type in _loaded_models:
        return _loaded_models[model_type]

    if model_type not in MODEL_FILES:
        raise ValueError(
            f"Type de modèle invalide: '{model_type}'. "
            f"Types disponibles: {', '.join(MODEL_FILES.keys())}"
        )

    model_filename = MODEL_FILES[model_type]
    model_path = MODELS_DIR / f"{model_filename}.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Le modèle '{model_type}' n'a pas été entraîné. "
            f"Fichier attendu: {model_path}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = MODEL_NUM_CLASSES[model_type]
    class_names = MODEL_CLASSES[model_type]

    model = _create_model(num_classes)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    _loaded_models[model_type] = (model, class_names)
    return model, class_names


def is_model_available(model_type: str) -> bool:
    """Vérifie si un modèle entraîné est disponible sur le disque."""
    if model_type not in MODEL_FILES:
        return False
    model_path = MODELS_DIR / f"{MODEL_FILES[model_type]}.pth"
    return model_path.exists()


def get_model_classes(model_type: str) -> list[str]:
    """Retourne les noms de classes d'un modèle."""
    return MODEL_CLASSES.get(model_type, [])


def predict_from_image(image_bytes: bytes, model_type: str) -> dict:
    """Effectue une prédiction à partir des octets d'une image.

    Args:
        image_bytes: Contenu binaire de l'image.
        model_type: Type de modèle à utiliser.

    Returns:
        Dictionnaire avec prediction, confidence, probabilities.
    """
    model, class_names = load_model(model_type)
    device = next(model.parameters()).device

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs[0], dim=0)

    probabilities = {
        class_names[i]: round(probs[i].item(), 4) for i in range(len(class_names))
    }

    max_idx = probs.argmax().item()
    prediction = class_names[max_idx]
    confidence = round(probs[max_idx].item(), 4)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
    }
