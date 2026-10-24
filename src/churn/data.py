"""Download and clean the IBM Telco churn dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from churn.config import DATA_URL, ID_COLUMN, RAW_DATA_PATH, TARGET

logger = logging.getLogger(__name__)


def download_data(url: str = DATA_URL, dest: Path = RAW_DATA_PATH, force: bool = False) -> Path:
    """Download the CSV unless it is already cached."""
    if dest.exists() and not force:
        logger.info("Using cached dataset at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix column types and encode the target.

    TotalCharges is a text column and is blank for customers with tenure 0. They haven't been
    billed yet, so blank means 0 here, not missing.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", "0", regex=True))
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    if ID_COLUMN in df.columns:
        df = df.set_index(ID_COLUMN)
    return df


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset (downloads it on first use)."""
    if not path.exists():
        download_data(dest=path)
    return clean(pd.read_csv(path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = load_dataset()
    logger.info("Loaded %d customers, churn rate %.1f%%", len(data), 100 * data[TARGET].mean())
