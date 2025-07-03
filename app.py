from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os
# ✅ Charger le modèle binaire (2 classes : -1, +1)
with open("gold_cross_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ App FastAPI
app = FastAPI(
    title="Gold Price Direction Predictor",
    description="API to predict probability of gold price going up or down in next 12x5min candles.",
    version="1.0"
)

# ✅ Schéma d'entrée EXACT des features utilisées au training
class InputFeatures(BaseModel):
    rsi: float
    ema_9: float
    ema_21: float
    ema_distance: float
    macd_line: float
    atr: float
    volatility_close_std: float
    ema_9_slope: float
    ema_21_slope: float

@app.get("/")
def home():
    return {"message": "API is live. Use POST /predict with your features."}

@app.post("/predict")
def predict(features: InputFeatures):
    # ✅ Convertir en dataframe
    df_input = pd.DataFrame([features.dict()])

    # ✅ Assurer l'ordre des colonnes
    expected_cols = list(model.feature_names_in_)
    df_input = df_input[expected_cols]

    # ✅ Obtenir les probas pour les 2 classes [-1, +1]
    proba = model.predict_proba(df_input)[0]
    p_baisse = proba[0]
    p_hausse = proba[1]

    # ✅ Logique de signal
    if p_hausse > 0.5 and p_hausse > p_baisse:
        signal = "long"
    elif p_baisse > 0.5 and p_baisse > p_hausse:
        signal = "short"
    else:
        signal = "no_trade"

    # ✅ Retour structuré
    return {
        "signal": signal,
        "probabilities": {
            "-1 (BAISSE)": round(p_baisse * 100, 2),
            "+1 (HAUSSE)": round(p_hausse * 100, 2)
        }
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), reload=False)
