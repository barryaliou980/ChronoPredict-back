from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas import PatientInput, PredictionResult
from app.ml.preprocess import DISEASE_CLASSES

router = APIRouter()

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_model = None
_preprocessor = None


def get_model():
    global _model, _preprocessor
    if _model is None:
        model_path = MODELS_DIR / "best_model.joblib"
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"

        if not model_path.exists() or not preprocessor_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Le modèle n'est pas encore entraîné. Exécutez 'python -m app.ml.train' d'abord.",
            )

        _model = joblib.load(model_path)
        _preprocessor = joblib.load(preprocessor_path)

    return _model, _preprocessor


def determine_risk_level(probabilities: dict[str, float]) -> str:
    healthy_prob = probabilities.get("Healthy", 0)
    if healthy_prob >= 0.6:
        return "Low"
    elif healthy_prob >= 0.3:
        return "Medium"
    return "High"


@router.post("/predict", response_model=PredictionResult)
async def predict(patient: PatientInput):
    model, preprocessor = get_model()

    data = pd.DataFrame([patient.model_dump()])

    try:
        X = preprocessor.transform(data)
        probas = model.predict_proba(X)[0]

        classes = model.classes_
        probabilities = {
            DISEASE_CLASSES.get(int(cls), str(cls)): round(float(prob), 4)
            for cls, prob in zip(classes, probas)
        }

        predicted_class = int(classes[np.argmax(probas)])
        prediction = DISEASE_CLASSES.get(predicted_class, str(predicted_class))
        risk_level = determine_risk_level(probabilities)

        return PredictionResult(
            prediction=prediction,
            probabilities=probabilities,
            risk_level=risk_level,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")
