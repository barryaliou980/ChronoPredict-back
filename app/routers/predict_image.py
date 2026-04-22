"""Routeur FastAPI pour la prédiction de maladies par image.

Endpoints pour l'analyse d'images médicales (radiographie, lésions cutanées).
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, Form

from app.ml.predict_image import (
    DISEASE_TO_IMAGE_MODEL,
    MODEL_DESCRIPTIONS,
    MODEL_FILES,
    get_model_classes,
    is_model_available,
    predict_from_image,
)

router = APIRouter()


@router.post("/image")
async def predict_image(
    file: UploadFile = File(..., description="Image médicale à analyser"),
    model_type: str = Form(..., description="Type de modèle: chest_xray, skin_lesion"),
):
    """Effectue une prédiction de maladie à partir d'une image médicale."""
    # Valider le type de modèle
    if model_type not in MODEL_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Type de modèle invalide: '{model_type}'. "
            f"Types disponibles: {', '.join(MODEL_FILES.keys())}",
        )

    # Valider le fichier
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être une image (JPEG, PNG, etc.).",
        )

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire le fichier envoyé.")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Le fichier envoyé est vide.")

    try:
        result = predict_from_image(image_bytes, model_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "model_type": model_type,
        "description": MODEL_DESCRIPTIONS.get(model_type, ""),
    }


@router.get("/image/models")
async def list_image_models():
    """Retourne la liste des modèles d'image disponibles et leur statut."""
    models_list = []
    for model_type, description in MODEL_DESCRIPTIONS.items():
        models_list.append({
            "model_type": model_type,
            "name": model_type.replace("_", " ").title(),
            "description": description,
            "classes": get_model_classes(model_type),
            "is_available": is_model_available(model_type),
        })
    return models_list


@router.get("/image/mapping")
async def get_disease_image_mapping():
    """Retourne la correspondance entre maladies et modèles d'image."""
    return DISEASE_TO_IMAGE_MODEL
