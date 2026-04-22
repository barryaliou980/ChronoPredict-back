import torch
from utils.modele import creer_modele
from utils.data_loader import preparer_chargeur_donnees

# NOUVEAU : Importation des outils de statistiques
from sklearn.metrics import confusion_matrix, recall_score

chemin_test = 'data/chest_xray/chest_xray/test'
chemin_modele = 'models/modele_pneumonie.pth'

def evaluer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modele = creer_modele()
    modele.load_state_dict(torch.load(chemin_modele, weights_only=True))
    modele = modele.to(device)
    modele.eval() 
    
    _, chargeur_test = preparer_chargeur_donnees(chemin_test, batch_size=32)
    
    # NOUVEAU : Listes pour retenir toutes les réponses de l'examen
    vraies_etiquettes = []
    predictions_ia = []
    
    print("📝 Début de l'évaluation détaillée...")
    
    with torch.no_grad(): 
        for images, labels in chargeur_test:
            images, labels = images.to(device), labels.to(device)
            predictions = modele(images)
            _, reponse_choisie = torch.max(predictions, 1)
            
            # On stocke les résultats (on doit les ramener sur le CPU pour Scikit-learn)
            vraies_etiquettes.extend(labels.cpu().numpy())
            predictions_ia.extend(reponse_choisie.cpu().numpy())
            
    # Calcul des métriques médicales
    # Note : Dans notre dataset, 0 = Normal, 1 = Pneumonie
    matrice = confusion_matrix(vraies_etiquettes, predictions_ia)
    sensibilite = recall_score(vraies_etiquettes, predictions_ia)
    
    print("\n📊 Matrice de confusion :")
    print(matrice)
    print(f"\n🚨 Sensibilité (Détection des pneumonies) : {sensibilite * 100:.2f}%")

if __name__ == "__main__":
    evaluer()