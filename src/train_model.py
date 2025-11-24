import joblib
import xgboost as xgb

from config import ARTIFACTS_DIR, XGB_RANDOM_STATE, XGB_SCALE_POS_WEIGHT


def main():
    X_train = joblib.load(ARTIFACTS_DIR / "X_train.joblib")
    y_train = joblib.load(ARTIFACTS_DIR / "y_train.joblib")

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=XGB_RANDOM_STATE,
        scale_pos_weight=XGB_SCALE_POS_WEIGHT,
    )

    model.fit(X_train, y_train)

    model_path = ARTIFACTS_DIR / "model_xgb.joblib"
    joblib.dump(model, model_path)

    print(f"Modelo treinado e salvo em {model_path}")


if __name__ == "__main__":
    main()
