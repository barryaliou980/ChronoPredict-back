from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

<<<<<<< HEAD
from app.routers import health, predict, predict_image
=======
from app.routers import health, predict, predict_image, xray_predict
>>>>>>> develop

app = FastAPI(
    title="Chronic Disease Prediction API",
    description="API de prédiction de maladies chroniques par intelligence artificielle",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(predict.router, prefix="/api", tags=["Prediction"])
<<<<<<< HEAD
app.include_router(predict_image.router, prefix="/api", tags=["Image Prediction"])
=======
app.include_router(predict_image.router, prefix="/api/predict", tags=["Prediction (Image)"])
app.include_router(xray_predict.router, prefix="/api/predict", tags=["Prediction (Radiographie)"])
>>>>>>> develop
