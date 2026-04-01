"""Script d'entraînement des modèles de prédiction de maladies chroniques."""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from app.ml.preprocess import create_preprocessor, load_and_prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def find_dataset() -> str:
    """Find the diabetes CSV dataset in the data directory."""
    target = DATA_DIR / "diabetes_data.csv"
    if target.exists():
        return str(target)
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"Erreur: Aucun fichier CSV trouvé dans {DATA_DIR}")
        print("Téléchargez le dataset depuis:")
        print("https://www.kaggle.com/datasets/prosperchuks/health-dataset")
        print(f"et placez les fichiers CSV dans {DATA_DIR}/")
        sys.exit(1)
    return str(csv_files[0])


def train():
    """Train multiple models and save the best one."""
    print("=" * 60)
    print("  Entraînement des modèles de prédiction")
    print("=" * 60)

    # Load data
    csv_path = find_dataset()
    print(f"\nDataset: {csv_path}")

    X, y = load_and_prepare_data(csv_path)
    print(f"Échantillons: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {y.unique().tolist()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Create preprocessor and fit on training data
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    # Train and evaluate
    results = {}
    best_model = None
    best_score = 0
    best_name = ""

    print("\n" + "=" * 60)
    print("  Résultats")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        results[name] = {"accuracy": accuracy, "f1": f1}

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(classification_report(y_test, y_pred))

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

    # Summary table
    print("\n" + "=" * 60)
    print("  Comparaison des modèles")
    print("=" * 60)
    print(f"\n{'Modèle':<25} {'Accuracy':>10} {'F1-score':>10}")
    print("-" * 47)
    for name, metrics in results.items():
        marker = " *" if name == best_name else ""
        print(
            f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['f1']:>10.4f}{marker}"
        )
    print(f"\n* Meilleur modèle: {best_name}")

    # Save best model and preprocessor
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"\nModèle sauvegardé: {model_path}")
    print(f"Preprocessor sauvegardé: {preprocessor_path}")
    print("\nTerminé!")


if __name__ == "__main__":
    train()
