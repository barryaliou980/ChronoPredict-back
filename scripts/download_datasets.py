"""Script de téléchargement et d'organisation des datasets d'images médicales.

Prérequis:
    1. Installer l'API Kaggle : pip install kaggle
    2. Configurer les identifiants Kaggle :
       - Créer un compte sur https://www.kaggle.com
       - Aller dans Account > API > Create New Token
       - Placer le fichier kaggle.json dans ~/.kaggle/
       - chmod 600 ~/.kaggle/kaggle.json

Datasets téléchargés:
    1. APTOS 2019 - Rétinopathie diabétique
       https://www.kaggle.com/c/aptos2019-blindness-detection
    2. Chest X-Ray Pneumonia
       https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    3. HAM10000 - Lésions cutanées
       https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Usage:
    cd backend
    python scripts/download_datasets.py

Structure attendue après téléchargement:
    backend/data/
    ├── aptos2019/
    │   └── train/
    │       ├── 0/   (No DR)
    │       ├── 1/   (Mild)
    │       ├── 2/   (Moderate)
    │       ├── 3/   (Severe)
    │       └── 4/   (Proliferative DR)
    ├── chest_xray/
    │   └── train/
    │       ├── NORMAL/
    │       └── PNEUMONIA/
    └── ham10000/
        └── train/
            ├── akiec/  (Actinic keratoses)
            ├── bcc/    (Basal cell carcinoma)
            ├── bkl/    (Benign keratosis)
            ├── df/     (Dermatofibroma)
            ├── mel/    (Melanoma)
            ├── nv/     (Melanocytic nevi)
            └── vasc/   (Vascular lesions)
"""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def check_kaggle_installed() -> bool:
    """Vérifie que l'outil Kaggle est installé et configuré."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  Kaggle CLI détecté: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("  Erreur: Kaggle CLI non installé.")
    print("  Installez-le avec: pip install kaggle")
    print("  Puis configurez vos identifiants (voir les commentaires de ce script).")
    return False


def download_aptos2019() -> None:
    """Télécharge et organise le dataset APTOS 2019 (rétinopathie diabétique).

    Le dataset Kaggle contient des images avec un fichier CSV de labels.
    On les réorganise en sous-dossiers par classe (0-4).
    """
    print("\n" + "=" * 60)
    print("  Téléchargement - APTOS 2019 (Rétinopathie Diabétique)")
    print("=" * 60)

    dest_dir = DATA_DIR / "aptos2019"
    train_dir = dest_dir / "train"

    if train_dir.exists() and any(train_dir.iterdir()):
        print("  Dataset déjà présent, étape ignorée.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Télécharger depuis Kaggle
    print("  Téléchargement depuis Kaggle...")
    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "aptos2019-blindness-detection",
            "-p", str(dest_dir),
        ],
        check=True,
    )

    # Extraire l'archive
    zip_file = dest_dir / "aptos2019-blindness-detection.zip"
    if zip_file.exists():
        print("  Extraction...")
        shutil.unpack_archive(str(zip_file), str(dest_dir))
        zip_file.unlink()

    # Organiser en sous-dossiers par classe
    print("  Organisation des images par classe...")
    images_dir = dest_dir / "train_images"
    csv_path = dest_dir / "train.csv"

    if not csv_path.exists() or not images_dir.exists():
        print("  Erreur: structure inattendue après extraction.")
        print(f"  Contenu de {dest_dir}: {list(dest_dir.iterdir())}")
        return

    # Créer les sous-dossiers (0 à 4)
    for label in range(5):
        (train_dir / str(label)).mkdir(parents=True, exist_ok=True)

    # Lire le CSV et déplacer les images
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row["id_code"] + ".png"
            label = row["diagnosis"]
            src = images_dir / image_name
            dst = train_dir / label / image_name
            if src.exists():
                shutil.move(str(src), str(dst))

    # Nettoyer
    if images_dir.exists():
        shutil.rmtree(str(images_dir))

    print("  Dataset APTOS 2019 prêt!")


def download_chest_xray() -> None:
    """Télécharge le dataset Chest X-Ray Pneumonia.

    Ce dataset est déjà organisé en sous-dossiers train/val/test
    avec les classes NORMAL et PNEUMONIA.
    """
    print("\n" + "=" * 60)
    print("  Téléchargement - Chest X-Ray (Pneumonie)")
    print("=" * 60)

    dest_dir = DATA_DIR / "chest_xray"
    train_dir = dest_dir / "train"

    if train_dir.exists() and any(train_dir.iterdir()):
        print("  Dataset déjà présent, étape ignorée.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Télécharger depuis Kaggle
    print("  Téléchargement depuis Kaggle...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "paultimothymooney/chest-xray-pneumonia",
            "-p", str(dest_dir),
        ],
        check=True,
    )

    # Extraire l'archive
    zip_file = dest_dir / "chest-xray-pneumonia.zip"
    if zip_file.exists():
        print("  Extraction...")
        shutil.unpack_archive(str(zip_file), str(dest_dir))
        zip_file.unlink()

    # Le dataset Kaggle extrait souvent dans un sous-dossier chest_xray/
    nested_dir = dest_dir / "chest_xray"
    if nested_dir.exists() and nested_dir.is_dir():
        # Déplacer le contenu au bon niveau
        for item in nested_dir.iterdir():
            target = dest_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(dest_dir))
        shutil.rmtree(str(nested_dir))

    print("  Dataset Chest X-Ray prêt!")


def download_ham10000() -> None:
    """Télécharge et organise le dataset HAM10000 (lésions cutanées).

    Le dataset contient des images et un fichier CSV de métadonnées.
    On les réorganise en sous-dossiers par type de lésion.
    """
    print("\n" + "=" * 60)
    print("  Téléchargement - HAM10000 (Lésions Cutanées)")
    print("=" * 60)

    dest_dir = DATA_DIR / "ham10000"
    train_dir = dest_dir / "train"

    if train_dir.exists() and any(train_dir.iterdir()):
        print("  Dataset déjà présent, étape ignorée.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Télécharger depuis Kaggle
    print("  Téléchargement depuis Kaggle...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "kmader/skin-cancer-mnist-ham10000",
            "-p", str(dest_dir),
        ],
        check=True,
    )

    # Extraire l'archive
    zip_file = dest_dir / "skin-cancer-mnist-ham10000.zip"
    if zip_file.exists():
        print("  Extraction...")
        shutil.unpack_archive(str(zip_file), str(dest_dir))
        zip_file.unlink()

    # Organiser en sous-dossiers par type de lésion
    print("  Organisation des images par type de lésion...")

    # Les classes HAM10000
    lesion_types = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    for lesion in lesion_types:
        (train_dir / lesion).mkdir(parents=True, exist_ok=True)

    # Chercher le fichier CSV de métadonnées
    csv_path = None
    for candidate in [
        dest_dir / "HAM10000_metadata.csv",
        dest_dir / "HAM10000_metadata",
    ]:
        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        print("  Erreur: HAM10000_metadata.csv introuvable.")
        print(f"  Contenu de {dest_dir}: {list(dest_dir.iterdir())}")
        return

    # Trouver les répertoires contenant les images
    image_dirs = []
    for item in dest_dir.rglob("*.jpg"):
        parent = item.parent
        if parent not in image_dirs:
            image_dirs.append(parent)

    # Lire le CSV et déplacer les images
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            dx = row["dx"]  # type de lésion
            if dx not in lesion_types:
                continue

            # Chercher l'image dans les répertoires trouvés
            for img_dir in image_dirs:
                src = img_dir / f"{image_id}.jpg"
                if src.exists():
                    dst = train_dir / dx / f"{image_id}.jpg"
                    shutil.move(str(src), str(dst))
                    break

    print("  Dataset HAM10000 prêt!")


def download_all() -> None:
    """Télécharge et organise tous les datasets."""
    print("=" * 60)
    print("  Téléchargement de tous les datasets d'images médicales")
    print("=" * 60)

    if not check_kaggle_installed():
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_aptos2019()
    download_chest_xray()
    download_ham10000()

    print("\n" + "=" * 60)
    print("  Tous les datasets ont été téléchargés et organisés!")
    print("=" * 60)
    print(f"\n  Emplacement: {DATA_DIR}")
    print("\n  Prochaine étape: entraîner les modèles avec:")
    print("    cd backend && python -m app.ml.train_image")


if __name__ == "__main__":
    download_all()
