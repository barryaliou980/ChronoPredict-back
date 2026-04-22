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

<<<<<<< HEAD

# --- Schémas pour la prédiction par image ---


class ImagePredictionInput(BaseModel):
    model_type: str = Field(
        ...,
        description="Type de modèle: retinopathy, chest_xray, skin_lesion",
    )


class ImagePredictionResult(BaseModel):
    prediction: str = Field(..., description="Classe prédite par le modèle")
    confidence: float = Field(..., description="Confiance de la prédiction (0-1)")
    probabilities: dict[str, float] = Field(
        ..., description="Probabilités pour chaque classe"
    )
    model_type: str = Field(..., description="Type de modèle utilisé")
    description: str = Field(
        ..., description="Description lisible de l'analyse effectuée"
    )


class ImageModelInfo(BaseModel):
    model_type: str = Field(..., description="Identifiant du type de modèle")
    name: str = Field(..., description="Nom lisible du modèle")
    description: str = Field(..., description="Description du modèle")
    diseases: list[str] = Field(
        ..., description="Maladies que ce modèle peut confirmer"
    )
    classes: list[str] = Field(..., description="Classes de sortie du modèle")
    is_available: bool = Field(
        ..., description="Indique si le modèle entraîné est disponible"
    )


class ImageModelsResponse(BaseModel):
    models: list[ImageModelInfo] = Field(
        ..., description="Liste des modèles d'image disponibles"
    )


class DiseaseMappingResponse(BaseModel):
    mapping: dict[str, str] = Field(
        ..., description="Correspondance maladie -> type de modèle d'image"
    )
=======
class XRayPredictionResponse(BaseModel):
    nom_fichier: str = Field(..., description="Nom du fichier image analysé")
    diagnostic: str = Field(..., description="Résultat du diagnostic: Normal ou Pneumonie")
    confiance: str = Field(..., description="Pourcentage de certitude de l'IA (ex: 98.50%)")
>>>>>>> develop
