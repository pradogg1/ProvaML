import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remover coluna Time se existir
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    # (Opcional) normalizar Amount – por enquanto vamos deixar como está

    return df
