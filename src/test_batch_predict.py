from pathlib import Path

import pandas as pd
import pytest

from src.batch_predict import run_batch
from src.config import ARTIFACTS_DIR


# Se o modelo ainda não foi treinado, pulamos todos os testes deste arquivo.
if not (ARTIFACTS_DIR / "model_xgb.joblib").exists():
    pytest.skip(
        "Modelo não treinado. Rode 'poetry run python -m src.train_model' antes dos testes.",
        allow_module_level=True,
    )


def _make_valid_row():
    """Cria um dicionário com uma linha válida para Transaction."""
    data = {f"V{i}": float(i) for i in range(1, 29)}
    data["Amount"] = 123.45
    return data


def test_batch_predict_valid_csv(tmp_path):
    # Arrange: criar CSV válido temporário
    input_path = tmp_path / "valid.csv"
    output_path = tmp_path / "preds.csv"

    df = pd.DataFrame([_make_valid_row(), _make_valid_row()])
    df.to_csv(input_path, index=False)

    # Act: rodar o batch
    df_out = run_batch(input_path, output_path)

    # Assert: saída foi criada e tem colunas esperadas
    assert output_path.exists(), "Arquivo de saída não foi criado."

    df_loaded = pd.read_csv(output_path)

    # mesmo número de linhas
    assert len(df_loaded) == len(df_out) == 2

    # colunas probability e prediction existem
    assert "probability" in df_loaded.columns
    assert "prediction" in df_loaded.columns

    # prediction deve ser 0 ou 1
    assert set(df_loaded["prediction"].unique()).issubset({0, 1})


def test_batch_predict_invalid_csv_raises(tmp_path):
    # Arrange: CSV inválido (sem coluna Amount)
    input_path = tmp_path / "invalid.csv"
    output_path = tmp_path / "preds_invalid.csv"

    row = _make_valid_row()
    row.pop("Amount")  # remover Amount para ficar inválido
    df = pd.DataFrame([row])
    df.to_csv(input_path, index=False)

    # Act + Assert: deve levantar ValueError por falha de validação Pydantic
    with pytest.raises(ValueError):
        run_batch(input_path, output_path)
