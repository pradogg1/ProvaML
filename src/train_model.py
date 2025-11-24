import joblib
import xgboost as xgb

from src.config import ARTIFACTS_DIR, XGB_PARAMS


def main():
    X_train = joblib.load(ARTIFACTS_DIR / "X_train.joblib")
    y_train = joblib.load(ARTIFACTS_DIR / "y_train.joblib")

    # usa os parâmetros do params.yaml
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        **XGB_PARAMS,
    )

    model.fit(X_train, y_train)

    model_path = ARTIFACTS_DIR / "model_xgb.joblib"
    joblib.dump(model, model_path)

    print(f"Modelo treinado e salvo em {model_path}")


if __name__ == "__main__":
    main()
