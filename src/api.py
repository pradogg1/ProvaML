from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, THRESHOLD
from src.features import make_features
from src.schemas import Transaction

# Carrega o modelo na inicialização do app
model = joblib.load(ARTIFACTS_DIR / "model_xgb.joblib")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API para predição de fraude em transações de cartão de crédito",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    # Converte o objeto Pydantic para dict
    data = transaction.model_dump()

    # Cria um DataFrame com uma única linha
    df = pd.DataFrame([data])

    # Garante que as mesmas features do treino são aplicadas
    X = make_features(df)

    # Probabilidade da classe 1 (fraude)
    proba = model.predict_proba(X)[:, 1][0]
    prediction = int(proba > THRESHOLD)

    return {
        "prediction": prediction,          # 0 = normal, 1 = fraude
        "probability": float(proba),       # probabilidade da transação ser fraude
        "threshold": THRESHOLD,
    }
