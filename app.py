
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# ✅ Charger les modèles entraînés
with open("ict_classifier.pkl", "rb") as f:
    classifier_model = pickle.load(f)

with open("ict_tp_model.pkl", "rb") as f:
    tp_model = pickle.load(f)

with open("ict_sl_model.pkl", "rb") as f:
    sl_model = pickle.load(f)

# ✅ Créer l'application FastAPI
app = FastAPI(
    title="ICT Trading Signal API",
    description="API pour prédire direction, Take Profit et Stop Loss sur Gold",
    version="1.0"
)

# ✅ Définir le schéma des features d'entrée
class ICTFeatures(BaseModel):
    rsi: float
    ema_9: float
    ema_21: float
    macd_line: float

# ✅ Endpoint racine
@app.get("/")
def read_root():
    return {"message": "Welcome to ICT Trading Signal API!"}

# ✅ Endpoint de prédiction
@app.post("/predict")
def predict(features: ICTFeatures):
    # Convertir les données en DataFrame
    data = pd.DataFrame([features.dict()])

    # Prediction Classification
    pred_class = int(classifier_model.predict(data)[0])
    pred_proba = float(classifier_model.predict_proba(data)[0][1])

    # Prediction Take Profit
    predicted_tp = float(tp_model.predict(data)[0])

    # Prediction Stop Loss
    predicted_sl = float(sl_model.predict(data)[0])

    # Résultat complet
    result = {
        "prediction": pred_class,
        "probability_of_increase": round(pred_proba * 100, 2),
        "take_profit_suggestion": round(predicted_tp, 2),
        "stop_loss_suggestion": round(predicted_sl, 2)
    }

    return result
