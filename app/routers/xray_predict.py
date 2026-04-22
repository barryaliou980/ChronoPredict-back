from fastapi import APIRouter, UploadFile, File, HTTPException
import torch
from torchvision import transforms
from PIL import Image
import io

# Importations internes
from app.utils.modele import creer_modele
from app.schemas import XRayPredictionResponse

router = APIRouter()

# --- Chargement du Modèle ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modele = creer_modele()
modele.load_state_dict(torch.load('models/modele_pneumonie.pth', map_location=device, weights_only=True))
modele.to(device)
modele.eval()

# --- Transformations de l'image ---
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.post("/xray", response_model=XRayPredictionResponse)
async def predire_pneumonie(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        image_brute = await file.read()
        image = Image.open(io.BytesIO(image_brute)).convert('RGB')
        
        tenseur_image = transformations(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = modele(tenseur_image)
            _, reponse_choisie = torch.max(predictions, 1)
            
            probabilites = torch.nn.functional.softmax(predictions[0], dim=0)
            confiance = probabilites[reponse_choisie].item() * 100
            
        diagnostic = "Pneumonie" if reponse_choisie.item() == 1 else "Normal"
        
        return XRayPredictionResponse(
            nom_fichier=file.filename,
            diagnostic=diagnostic,
            confiance=f"{confiance:.2f}%"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse : {str(e)}")