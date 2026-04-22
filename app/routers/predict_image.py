"""Routeur FastAPI pour la prédiction de maladies par image.

Endpoints pour l'analyse d'images médicales (rétinopathie, radiographie, lésions cutanées).
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.ml.predict_image import (
    DISEASE_TO_IMAGE_MODEL,
    MODEL_DESCRIPTIONS,
    MODEL_FILES,
    get_model_classes,
    is_model_available,
    predict_from_image,
)
from app.schemas import (
    DiseaseMappingResponse,
    ImageModelInfo,
    ImageModelsResponse,
    ImagePredictionResult,
)

router = APIRouter()

# Noms lisibles des modèles
MODEL_NAMES = {
    "retinopathy": "Rétinopathie Diabétique (APTOS 2019)",
    "chest_xray": "Radiographie Thoracique (Pneumonie)",
    "skin_lesion": "Lésions Cutanées (HAM10000)",
}

# Maladies associées à chaque modèle
MODEL_DISEASES = {
    "retinopathy": ["Diabetes"],
    "chest_xray": ["Pneumonia"],
    "skin_lesion": ["Psoriasis", "Acne", "Impetigo", "Fungal infection"],
}

# Classes par défaut (quand le modèle n'est pas encore entraîné)
DEFAULT_CLASSES = {
    "retinopathy": ["0", "1", "2", "3", "4"],
    "chest_xray": ["NORMAL", "PNEUMONIA"],
    "skin_lesion": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
}


@router.post("/predict/image", response_model=ImagePredictionResult)
async def predict_image(
    file: UploadFile = File(..., description="Image médicale à analyser"),
    model_type: str = Form(
        ...,
        description="Type de modèle: retinopathy, chest_xray, skin_lesion",
    ),
):
    """Effectue une prédiction de maladie à partir d'une image médicale.

    - **file**: Image au format JPEG/PNG (radiographie, fond d'oeil, dermoscopie)
    - **model_type**: Type de modèle à utiliser pour l'analyse
    """
    # Valider le type de modèle
    if model_type not in MODEL_FILES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Type de modèle invalide: '{model_type}'. "
                f"Types disponibles: {list(MODEL_FILES.keys())}"
            ),
        )

    # Valider le type de fichier
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être une image (JPEG, PNG, etc.).",
        )

    # Lire le contenu de l'image
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Impossible de lire le fichier envoyé.",
        )

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Le fichier envoyé est vide.",
        )

    # Effectuer la prédiction
    try:
        result = predict_from_image(image_bytes, model_type)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}",
        )

    return ImagePredictionResult(
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_type=model_type,
        description=MODEL_DESCRIPTIONS.get(model_type, ""),
    )


@router.get("/predict/image/models", response_model=ImageModelsResponse)
async def get_image_models():
    """Retourne la liste des modèles d'image disponibles et leur statut."""
    models_info = []

    for model_type in MODEL_FILES:
        available = is_model_available(model_type)
        classes = get_model_classes(model_type) if available else DEFAULT_CLASSES.get(model_type, [])

        models_info.append(
            ImageModelInfo(
                model_type=model_type,
                name=MODEL_NAMES.get(model_type, model_type),
                description=MODEL_DESCRIPTIONS.get(model_type, ""),
                diseases=MODEL_DISEASES.get(model_type, []),
                classes=classes,
                is_available=available,
            )
        )

    return ImageModelsResponse(models=models_info)


@router.get("/predict/image/mapping", response_model=DiseaseMappingResponse)
async def get_disease_mapping():
    """Retourne la correspondance entre maladies et modèles d'image.

    Indique quel modèle d'image peut confirmer chaque maladie prédite par symptômes.
    """
    return DiseaseMappingResponse(mapping=DISEASE_TO_IMAGE_MODEL)
