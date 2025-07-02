from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# ✅ Charger modèle
with open("gold_entry_signal_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ App FastAPI
app = FastAPI(
    title="Gold Entry Signal Predictor",
    description="API for predicting trading signals based on EMA/RSI/MACD/ATR/Volatility",
    version="1.0"
)

# ✅ Schéma entrée
class InputFeatures(BaseModel):
    rsi: float
    ema_9: float
    ema_21: float
    macd_line: float
    atr: float
    volatility_close_std: float

@app.get("/")
def home():
    return {"message": "API is live. Use /predict endpoint."}

@app.post("/predict")
def predict(features: InputFeatures):
    # ✅ Convertir en dataframe
    df_input = pd.DataFrame([features.dict()])

    # ✅ S'assurer que l'ordre des colonnes correspond
    expected_cols = list(model.feature_names_in_)
    df_input = df_input[expected_cols]

    # ✅ Obtenir les probas pour chaque classe
    proba = model.predict_proba(df_input)[0]
    p_minus1 = proba[0]
    p_0 = proba[1]
    p_1 = proba[2]

    # ✅ Logique de signal
    if p_1 > 0.5 and p_1 > p_minus1:
        signal = "long"
    elif p_minus1 > 0.5 and p_minus1 > p_1:
        signal = "short"
    else:
        signal = "no_trade"

    # ✅ Retour structuré
    return {
        "signal": signal,
        "probabilities": {
            "-1 (short)": round(p_minus1 * 100, 2),
            "0 (no cross)": round(p_0 * 100, 2),
            "+1 (long)": round(p_1 * 100, 2)
        }
    }
