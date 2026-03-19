"""
data_preprocessing.py
---------------------
Milestone 1: Load raw CSV, clean column names, drop ID columns.
Returns a clean DataFrame ready for feature engineering.
"""

import pandas as pd
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw AI4I 2020 dataset from disk."""
    logger.info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Drop irrelevant ID columns and rename for cleaner coding.

    Parameters
    ----------
    df     : Raw DataFrame from load_data()
    config : Loaded YAML config (features section used)

    Returns
    -------
    Cleaned DataFrame
    """
    # Drop non-predictive ID columns
    drop_cols = [c for c in config["features"]["drop_columns"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    logger.info(f"Dropped columns: {drop_cols}")

    # Rename columns to remove units like [K], [rpm] etc.
    rename_map = config["features"]["rename_map"]
    df = df.rename(columns=rename_map)
    logger.info(f"Columns after rename: {list(df.columns)}")

    # Sanity check — no nulls in core sensor columns
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values detected:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("No null values found.")

    return df


def run_preprocessing(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """
    Full Milestone 1 pipeline: load → clean.
    Returns a clean DataFrame.
    """
    config = load_config(config_path)
    df_raw = load_data(config["data"]["raw_path"])
    df_clean = clean_data(df_raw, config)
    return df_clean


if __name__ == "__main__":
    df = run_preprocessing()
    print(df.head())
    print(df.dtypes)
