from pydantic import BaseModel, Field


class SymptomsInput(BaseModel):
    symptoms: list[str] = Field(
        ...,
        description="Liste des symptômes du patient",
        min_length=1,
        examples=[["headache", "chest_pain", "dizziness", "sweating"]],
    )


class PredictionResult(BaseModel):
    prediction: str = Field(..., description="Catégorie prédite")
    probabilities: dict[str, float] = Field(
        ..., description="Probabilités par catégorie"
    )
    risk_level: str = Field(..., description="Niveau de risque: Low, Medium, High")


class SymptomsListResponse(BaseModel):
    symptoms: list[str] = Field(..., description="Liste de tous les symptômes disponibles")

class XRayPredictionResponse(BaseModel):
    nom_fichier: str = Field(..., description="Nom du fichier image analysé")
    diagnostic: str = Field(..., description="Résultat du diagnostic: Normal ou Pneumonie")
    confiance: str = Field(..., description="Pourcentage de certitude de l'IA (ex: 98.50%)")
