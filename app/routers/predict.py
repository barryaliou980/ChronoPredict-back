from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import SymptomsInput, PredictionResult, SymptomsListResponse

router = APIRouter()

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_model = None
_label_encoder = None
_symptoms: list[str] = []


def get_model():
    global _model, _label_encoder, _symptoms
    if _model is None:
        model_path = MODELS_DIR / "best_model.joblib"
        le_path = MODELS_DIR / "label_encoder.joblib"
        symptoms_path = MODELS_DIR / "symptoms.joblib"

        if not model_path.exists() or not le_path.exists() or not symptoms_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Le modèle n'est pas encore entraîné. Exécutez 'python -m app.ml.train' d'abord.",
            )

        _model = joblib.load(model_path)
        _label_encoder = joblib.load(le_path)
        _symptoms = joblib.load(symptoms_path)

    return _model, _label_encoder, _symptoms


def determine_risk_level(top_probability: float) -> str:
    if top_probability >= 0.7:
        return "High"
    elif top_probability >= 0.4:
        return "Medium"
    return "Low"


@router.get("/symptoms", response_model=SymptomsListResponse)
async def get_symptoms():
    """Return the list of all available symptoms."""
    _, _, symptoms = get_model()
    return SymptomsListResponse(symptoms=symptoms)


@router.post("/predict", response_model=PredictionResult)
async def predict(input_data: SymptomsInput):
    model, label_encoder, symptoms = get_model()

    # Build feature vector
    feature_vector = {s: 0 for s in symptoms}
    for symptom in input_data.symptoms:
        if symptom in feature_vector:
            feature_vector[symptom] = 1

    data = pd.DataFrame([feature_vector])

    try:
        probas = model.predict_proba(data)[0]
        classes = label_encoder.classes_

        # Top 5 most probable diseases
        top_indices = np.argsort(probas)[::-1][:5]
        probabilities = {
            str(classes[i]): round(float(probas[i]), 4) for i in top_indices
        }

        predicted = str(classes[top_indices[0]])
        risk_level = determine_risk_level(float(probas[top_indices[0]]))

        return PredictionResult(
            prediction=predicted,
            probabilities=probabilities,
            risk_level=risk_level,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")
