import torch
import torch.nn as nn
import torch.optim as optim

# We absolutely need the data_loader to read and transform the images!
from utils.model_skincancer import create_skincancer_model 
from utils.data_loader_skincancer import prepare_skincancer_loaders

def train_skin_cancer():
    print(f"\n{'='*50}")
    print("🚀 Starting training for: Skin Cancer (HAM10000)")
    print(f"{'='*50}")
    
    # Configuration
    train_path = "data/skin_cancer_ready/train"
    num_classes = 7  # 7 specific skin lesion categories
    save_path = "models/skin_cancer_model.pth"
    epochs = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Using device: {device}")
    
    # 1. Preparing the DataLoaders
    print("📂 Loading images and applying transformations...")
    train_loader, test_loader = prepare_skincancer_loaders(train_path, batch_size=32)
    
    # 2. Initializing the model (with 7 output classes)
    print("🧠 Initializing ResNet model...")
    model = create_skincancer_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    print("🔥 Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # 4. Save the model weights
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved successfully at {save_path}")

if __name__ == "__main__":
    train_skin_cancer()