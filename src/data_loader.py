"""
Utility functions for loading project datasets.

Note: The main analysis pipeline is implemented in the Jupyter notebooks (01–08).
This module provides lightweight helpers to standardize file paths and I/O.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "prices_2018_2024.parquet")


def load_log_returns() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "log_returns_2018_2024.parquet")


def load_model_features() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "model_features_2018_2024.parquet")