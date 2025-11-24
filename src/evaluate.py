import json
import joblib
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from config import ARTIFACTS_DIR, THRESHOLD


def main():
    X_test = joblib.load(ARTIFACTS_DIR / "X_test.joblib")
    y_test = joblib.load(ARTIFACTS_DIR / "y_test.joblib")
    model = joblib.load(ARTIFACTS_DIR / "model_xgb.joblib")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > THRESHOLD).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "threshold": THRESHOLD,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "classification_report": report,
    }

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)

    print("Métricas de avaliação:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("Matriz de confusão:")
    print(cm)
    print(f"Métricas salvas em {metrics_path}")


if __name__ == "__main__":
    main()
