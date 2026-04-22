import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

print("🔄 Début de la préparation du dataset HAM10000...")

# Chemins de base (basés sur ta capture d'écran)
dossier_base = "data/skin_cancer"
fichier_csv = os.path.join(dossier_base, "HAM10000_metadata.csv")
dossiers_images = [
    os.path.join(dossier_base, "HAM10000_images_part_1"),
    os.path.join(dossier_base, "HAM10000_images_part_2")
]
dossier_sortie = "data/skin_cancer_ready"

# 1. Lecture du fichier CSV
print("📊 Lecture des métadonnées...")
df = pd.read_csv(fichier_csv)

# 2. Séparation mathématique (Train: 80%, Val: 10%, Test: 10%)
# stratify=df['dx'] est crucial en médecine : ça garantit qu'on a le même pourcentage 
# de chaque type de cancer dans nos 3 dossiers finaux.
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

splits = {
    'train': train_df,
    'val': val_df,
    'test': test_df
}

# 3. Création des dossiers et copie des images
print("📂 Création de l'arborescence et copie des fichiers (cela peut prendre quelques minutes)...")
for split_nom, split_data in splits.items():
    for index, row in split_data.iterrows():
        image_id = row['image_id']
        classe = row['dx'] # 'dx' est la colonne du diagnostic dans HAM10000
        nom_fichier = f"{image_id}.jpg"
        
        # Création du dossier cible (ex: data/skin_cancer_ready/train/mel)
        dossier_dest = os.path.join(dossier_sortie, split_nom, classe)
        os.makedirs(dossier_dest, exist_ok=True)
        chemin_dest = os.path.join(dossier_dest, nom_fichier)
        
        # Recherche de l'image dans part_1 ou part_2
        chemin_source = None
        for dossier in dossiers_images:
            chemin_potentiel = os.path.join(dossier, nom_fichier)
            if os.path.exists(chemin_potentiel):
                chemin_source = chemin_potentiel
                break
        
        # Copie physique du fichier
        if chemin_source:
            shutil.copy2(chemin_source, chemin_dest)
        else:
            print(f"⚠️ Image introuvable : {nom_fichier}")

print(f"✅ Terminé ! Ton dataset est trié et prêt à être entraîné dans : {dossier_sortie}")