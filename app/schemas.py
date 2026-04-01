from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=13, description="Catégorie d'âge (1=18-24, 2=25-29, ..., 13=80+)")
    sex: float = Field(..., ge=0, le=1, description="Sexe: 0 = Femme, 1 = Homme")
    high_chol: float = Field(..., ge=0, le=1, description="Cholestérol élevé: 0 = Non, 1 = Oui")
    chol_check: float = Field(..., ge=0, le=1, description="Contrôle cholestérol dans les 5 ans: 0 = Non, 1 = Oui")
    bmi: float = Field(..., ge=10, le=100, description="Indice de masse corporelle")
    smoker: float = Field(..., ge=0, le=1, description="Fumeur: 0 = Non, 1 = Oui")
    phys_activity: float = Field(..., ge=0, le=1, description="Activité physique dans les 30 derniers jours: 0 = Non, 1 = Oui")
    fruits: float = Field(..., ge=0, le=1, description="Consomme des fruits quotidiennement: 0 = Non, 1 = Oui")
    veggies: float = Field(..., ge=0, le=1, description="Consomme des légumes quotidiennement: 0 = Non, 1 = Oui")
    hvy_alcohol_consump: float = Field(..., ge=0, le=1, description="Forte consommation d'alcool: 0 = Non, 1 = Oui")
    gen_hlth: float = Field(..., ge=1, le=5, description="Santé générale: 1 = Excellent, 5 = Mauvais")
    ment_hlth: float = Field(..., ge=0, le=30, description="Jours de mauvaise santé mentale (0-30)")
    phys_hlth: float = Field(..., ge=0, le=30, description="Jours de mauvaise santé physique (0-30)")
    diff_walk: float = Field(..., ge=0, le=1, description="Difficulté à marcher: 0 = Non, 1 = Oui")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 7.0,
                    "sex": 1.0,
                    "high_chol": 1.0,
                    "chol_check": 1.0,
                    "bmi": 28.0,
                    "smoker": 0.0,
                    "phys_activity": 1.0,
                    "fruits": 1.0,
                    "veggies": 1.0,
                    "hvy_alcohol_consump": 0.0,
                    "gen_hlth": 3.0,
                    "ment_hlth": 5.0,
                    "phys_hlth": 10.0,
                    "diff_walk": 0.0,
                }
            ]
        }
    }


class PredictionResult(BaseModel):
    prediction: str = Field(..., description="Catégorie prédite")
    probabilities: dict[str, float] = Field(
        ..., description="Probabilités par catégorie"
    )
    risk_level: str = Field(..., description="Niveau de risque: Low, Medium, High")
