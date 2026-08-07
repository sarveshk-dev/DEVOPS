from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import joblib
Vmodel = joblib.load("model.pkl")
model = joblib.load("model.pkl")
app = FastAPI()
class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
@app.get("/")
def home():
    return {
        "message": "API is Running Successfully"
    }
    
@app.post("/predict")
def predict(data: HouseData):

    features = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    prediction = model.predict(features)

    return {
        "Predicted House Price": float(prediction[0])
    }
    
    
