import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)

from src.config import ARTIFACTS_DIR, THRESHOLD


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = joblib.load(ARTIFACTS_DIR / "X_test.joblib")
    y_test = joblib.load(ARTIFACTS_DIR / "y_test.joblib")
    model = joblib.load(ARTIFACTS_DIR / "model_xgb.joblib")

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    graph_path = ARTIFACTS_DIR / "graphs.png"

    # Probabilidades da classe positiva (fraude)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > THRESHOLD).astype(int)

    # Métricas agregadas
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Precision–Recall Curve (usando todos os thresholds)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    # Salva os pontos da curva em um TSV para o DVC (dvc plots diff)
    prc_path = ARTIFACTS_DIR / "prc.tsv"
    prc_df = pd.DataFrame(
        {
            "recall": recalls,
            "precision": precisions,
        }
    )
    prc_df.to_csv(prc_path, sep="\t", index=False)


    metrics = {
        "threshold": THRESHOLD,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "classification_report": report,
    }

    # Salva métricas em JSON (para DVC e comparação numérica)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)

    print("Métricas de avaliação (com threshold atual):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print(f"Average Precision (PRC): {avg_precision:.4f}")
    print("Matriz de confusão:")
    print(cm)
    print(f"Métricas salvas em {metrics_path}")

    # Gera o gráfico da Precision–Recall Curve
    fig, ax = plt.subplots()

    # curva PR (step é comum em PRC)
    ax.step(recalls, precisions, where="post", label=f"PR curve (AP = {avg_precision:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")

    fig.tight_layout()
    plt.savefig(graph_path)
    plt.close(fig)

    print(f"Gráfico Precision–Recall Curve salvo em {graph_path}")


if __name__ == "__main__":
    main()
