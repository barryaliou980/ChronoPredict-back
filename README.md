# Backend - Chronic Disease Prediction API

API FastAPI pour la prédiction de maladies chroniques avec des modèles de machine learning.

## Prérequis

- Python 3.11+

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Entraînement du modèle

Avant la première utilisation, entraîner le modèle ML :

```bash
python -m app.ml.train
```

Les modèles entraînés seront sauvegardés dans le dossier `models/`.

## Lancement

```bash
uvicorn app.main:app --reload
```

Le serveur démarre sur http://localhost:8000.

## Documentation API

Une fois le serveur lancé :

- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc

## Lancement avec Docker

```bash
docker build -t chronic-disease-backend .
docker run -p 8000:8000 chronic-disease-backend
```

## Structure

```
backend/
├── app/
│   ├── main.py        # Point d'entrée FastAPI
│   ├── routers/       # Endpoints API
│   ├── ml/            # Entraînement et préprocessing ML
│   └── schemas.py     # Modèles Pydantic
├── data/              # Dataset
├── models/            # Modèles entraînés (.joblib)
├── tests/             # Tests
├── requirements.txt
└── Dockerfile
```

## Dépendances principales

- FastAPI 0.115
- Scikit-learn 1.5
- Pandas 2.2
- Uvicorn 0.30
