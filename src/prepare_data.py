import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_RAW, ARTIFACTS_DIR, RANDOM_STATE_SPLIT, TEST_SIZE
from features import make_features


def main():
    print(f"Lendo dados de {DATA_RAW}...")
    df = pd.read_csv(DATA_RAW)

    # Separar target
    y = df["Class"]
    X = df.drop(columns=["Class"])

    # Aplicar features
    X = make_features(X)

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE_SPLIT,
        stratify=y,
    )

    # Criar pasta artifacts se não existir
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Salvar arquivos
    joblib.dump(X_train, ARTIFACTS_DIR / "X_train.joblib")
    joblib.dump(X_test, ARTIFACTS_DIR / "X_test.joblib")
    joblib.dump(y_train, ARTIFACTS_DIR / "y_train.joblib")
    joblib.dump(y_test, ARTIFACTS_DIR / "y_test.joblib")

    print("Dados preparados.")
    print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}")


if __name__ == "__main__":
    main()
