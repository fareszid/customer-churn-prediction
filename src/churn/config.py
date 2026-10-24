"""Paths, constants and campaign assumptions."""

import os
from pathlib import Path


def _project_root() -> Path:
    """Use CHURN_PROJECT_ROOT if set, otherwise walk up to the folder with pyproject.toml."""
    if os.getenv("CHURN_PROJECT_ROOT"):
        return Path(os.environ["CHURN_PROJECT_ROOT"])
    for folder in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    return Path.cwd()


PROJECT_ROOT = _project_root()

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", PROJECT_ROOT / "models" / "churn_model.joblib"))
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Retention campaign assumptions. They are not in the data: change them to test other scenarios.
OFFER_COST = 50.0  # $ per contacted customer (outreach + discount)
SAVE_RATE = 0.35  # share of churners who stay after getting the offer
MONTHS_RETAINED = 12  # months of revenue kept when a churner is saved
