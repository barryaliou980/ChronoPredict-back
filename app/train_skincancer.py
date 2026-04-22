import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from utils.model_skincancer import create_skincancer_model
from utils.data_loader_skincancer import prepare_skincancer_loaders

def train_skincancer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 6
    
    # 1. Chargement des données
    train_loader, test_loader = prepare_skincancer_loaders("data/skin_cancer_ready/train", batch_size=32)
    
    # 2. Initialisation
    model = create_skincancer_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"🚀 Entraînement SKIN CANCER (6 époques) sur {device}...")

    # 3. Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}] terminée.")

    # 4. ÉVALUATION POUR LE TABLEAU
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\n📊 --- MÉTRIQUES SKIN CANCER ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.2f}")
    # On utilise les noms de classes détectés automatiquement
    print(classification_report(y_true, y_pred, target_names=train_loader.dataset.classes))
    
    torch.save(model.state_dict(), "models/skin_cancer_model_6epochs.pth")

if __name__ == "__main__":
    train_skincancer()