import argparse
from pathlib import Path

import joblib
import pandas as pd
from pydantic import ValidationError

from src.config import ARTIFACTS_DIR, THRESHOLD
from src.features import make_features
from src.schemas import Transaction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch prediction for credit card fraud detection."
    )
    parser.add_argument("--input", "-i", required=True, help="CSV de entrada.")
    parser.add_argument("--output", "-o", required=True, help="CSV de saída.")
    return parser.parse_args()


def validate_csv_with_pydantic(df: pd.DataFrame):
    """Valida cada linha do CSV usando o modelo Pydantic Transaction.
    Retorna uma lista de erros."""
    errors = []

    for idx, row in df.iterrows():
        data = row.to_dict()
        try:
            Transaction(**data)
        except ValidationError as e:
            errors.append({"row": idx, "errors": e.errors()})

    return errors


def run_batch(input_path: Path, output_path: Path):
    """Função principal do batch, reutilizável em testes."""
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    print(f"Lendo arquivo: {input_path}")
    df = pd.read_csv(input_path)

    # Remove coluna Class, se existir
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # Validação Pydantic linha a linha
    print("Validando estrutura do CSV com Pydantic...")
    errors = validate_csv_with_pydantic(df)

    if errors:
        print("⚠️ Erros encontrados no CSV:")
        for err in errors[:10]:  # mostra só os 10 primeiros
            print(f"Linha {err['row']}: {err['errors']}")
        raise ValueError(
            f"CSV inválido. Total de linhas com erro: {len(errors)}. Corrija o arquivo antes do batch."
        )

    print("CSV validado com sucesso ✔️")

    # Feature engineering
    X = make_features(df)

    # Carregar modelo
    model_path = ARTIFACTS_DIR / "model_xgb.joblib"
    print(f"Carregando modelo de {model_path}...")
    model = joblib.load(model_path)

    # Probabilidades e predições
    proba = model.predict_proba(X)[:, 1]
    preds = (proba > THRESHOLD).astype(int)

    # Montar saída
    df_out = df.copy()
    df_out["probability"] = proba
    df_out["prediction"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print(f"Previsões salvas em {output_path}")
    print(
        "Resumo das classes previstas (0=normal, 1=fraude):",
        df_out["prediction"].value_counts().to_dict(),
    )

    return df_out  # útil para testes


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    run_batch(input_path, output_path)


if __name__ == "__main__":
    main()
