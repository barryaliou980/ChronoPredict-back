"""Script d'entraînement des modèles de prédiction de maladies chroniques."""

import sys
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.ml.preprocess import load_and_prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def train():
    """Train multiple models and save the best one."""
    print("=" * 60)
    print("  Entraînement des modèles de prédiction")
    print("=" * 60)

    train_path = DATA_DIR / "Training.csv"
    test_path = DATA_DIR / "Testing.csv"

    if not train_path.exists():
        print(f"Erreur: {train_path} introuvable")
        print("Téléchargez le dataset depuis:")
        print("https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning")
        print(f"et placez les fichiers CSV dans {DATA_DIR}/")
        sys.exit(1)

    X_train_full, y_train_full, label_encoder, X_test_provided, y_test_provided = (
        load_and_prepare_data(
            str(train_path), str(test_path) if test_path.exists() else None
        )
    )

    print(f"\nDataset: {train_path}")
    print(f"Échantillons: {len(X_train_full)}")
    print(f"Symptômes: {X_train_full.shape[1]}")
    print(f"Maladies: {len(label_encoder.classes_)}")

    # Split training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f"\nTrain: {len(X_train)} | Validation: {len(X_val)}")

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
    }

    # Train and evaluate
    results = {}
    best_model = None
    best_score = 0
    best_name = ""

    print("\n" + "=" * 60)
    print("  Résultats (Validation)")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")

        results[name] = {"accuracy": accuracy, "f1": f1}

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")

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

    # Retrain best model on full training data
    print(f"\nRé-entraînement de {best_name} sur toutes les données...")
    best_model.fit(X_train_full, y_train_full)

    # Test on provided test set
    if X_test_provided is not None and y_test_provided is not None:
        y_test_pred = best_model.predict(X_test_provided)
        test_acc = accuracy_score(y_test_provided, y_test_pred)
        print(f"Accuracy sur le jeu de test: {test_acc:.4f}")

    # Save model, label encoder, and symptom list
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(list(X_train_full.columns), MODELS_DIR / "symptoms.joblib")

    print(f"\nModèle sauvegardé: {MODELS_DIR / 'best_model.joblib'}")
    print(f"Label encoder sauvegardé: {MODELS_DIR / 'label_encoder.joblib'}")
    print(f"Symptômes sauvegardés: {MODELS_DIR / 'symptoms.joblib'}")
    print("\nTerminé!")


if __name__ == "__main__":
    train()
