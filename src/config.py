from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE_SPLIT = 183212
TEST_SIZE = 0.2

XGB_RANDOM_STATE = 37
XGB_SCALE_POS_WEIGHT = 300

THRESHOLD = 0.3  # limiar para classificar como fraude
