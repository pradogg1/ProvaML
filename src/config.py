from pathlib import Path
import yaml

# raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# arquivo de parâmetros
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

with PARAMS_FILE.open() as f:
    PARAMS = yaml.safe_load(f)

# hiperparâmetros do modelo
XGB_PARAMS = PARAMS["train"]

# threshold usado em avaliação/API/batch
THRESHOLD = PARAMS["eval"]["threshold"]

# split
RANDOM_STATE_SPLIT = 183212
TEST_SIZE = 0.2
