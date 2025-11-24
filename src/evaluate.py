import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import ARTIFACTS_DIR, THRESHOLD


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = joblib.load(ARTIFACTS_DIR / "X_test.joblib")
    y_test = joblib.load(ARTIFACTS_DIR / "y_test.joblib")
    model = joblib.load(ARTIFACTS_DIR / "model_xgb.joblib")

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    prev_metrics_path = ARTIFACTS_DIR / "metrics_prev.json"
    graph_path = ARTIFACTS_DIR / "graphs.png"

    # 1) Carrega métricas antigas, se existirem
    old_metrics = None
    if metrics_path.exists():
        with metrics_path.open() as f:
            old_metrics = json.load(f)

    # 2) Calcula as novas métricas
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

    # 3) Salva SEMPRE metrics.json (novo)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)

    print("Métricas novas:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("Matriz de confusão:")
    print(cm)
    print(f"Métricas salvas em {metrics_path}")

    # 4) Salva SEMPRE metrics_prev.json
    #    - se havia métricas antigas, guarda elas
    #    - senão, usa as novas como "primeira versão"
    prev_data = old_metrics if old_metrics is not None else metrics
    with prev_metrics_path.open("w") as f:
        json.dump(prev_data, f, indent=4)

    # 5) Gera SEMPRE um gráfico graphs.png
    labels = ["precision", "recall", "f1", "roc_auc"]

    old_vals = [
        prev_data.get(k, 0.0) for k in labels
    ]  # se for primeira vez, prev_data == metrics
    new_vals = [metrics[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    if old_metrics is not None:
        # temos modelo antigo: gráfico comparando antigo vs novo
        ax.bar(x - width / 2, old_vals, width, label="antigo")
        ax.bar(x + width / 2, new_vals, width, label="novo")
        ax.set_title("Comparação de desempenho – modelo antigo vs novo")
    else:
        # primeira vez: só barra do modelo atual
        ax.bar(x, new_vals, width, label="atual")
        ax.set_title("Desempenho do modelo atual")

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    plt.savefig(graph_path)
    plt.close(fig)

    print(f"Gráfico de desempenho salvo em {graph_path}")


if __name__ == "__main__":
    main()
