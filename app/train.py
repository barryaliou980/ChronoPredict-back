import torch
import torch.nn as nn
import torch.optim as optim
import os

# Importation de nos propres fichiers
from utils.data_loader import preparer_chargeur_donnees
from utils.modele import creer_modele

chemin_donnees = 'data/chest_xray/chest_xray/train'

def entrainer(epochs=3):
    # Détection de la carte graphique (CUDA pour les cartes NVIDIA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Matériel utilisé pour l'entraînement : {device}")
    print("🚀 Préparation de l'entraînement...")
    dataset, chargeur = preparer_chargeur_donnees(chemin_donnees, batch_size=32)
    modele = creer_modele()
    modele.to(device)

    # Le critère mesure à quel point le modèle se trompe
    critere = nn.CrossEntropyLoss() 
    # L'optimiseur modifie les poids pour corriger les erreurs
    optimiseur = optim.Adam(modele.fc.parameters(), lr=0.001)

    print("🧠 Début de l'apprentissage...")
    for epoque in range(epochs):
        for images, labels in chargeur:
            images, labels = images.to(device), labels.to(device) # On envoie les données sur le même matériel que le modèle
            optimiseur.zero_grad()           # 1. On efface la mémoire de la dernière étape
            predictions = modele(images)     # 2. Le modèle tente de deviner
            perte = critere(predictions, labels) # 3. On calcule la note (l'erreur)
            perte.backward()                 # 4. On analyse ce qui s'est mal passé
            optimiseur.step()                # 5. On corrige le modèle
            
        print(f"Époque {epoque+1}/{epochs} terminée. Erreur : {perte.item():.4f}")
    
    # Sauvegarde finale
    os.makedirs('models', exist_ok=True)
    torch.save(modele.state_dict(), 'models/modele_pneumonie.pth')
    print("✅ Modèle sauvegardé dans models/modele_pneumonie.pth")

if __name__ == "__main__":
    entrainer()