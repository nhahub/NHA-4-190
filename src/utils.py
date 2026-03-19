"""
utils.py
--------
Shared helpers: config loading, logging setup, model persistence.
"""

import os
import logging
import yaml
import joblib


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with a consistent format.
    Call once per module: logger = get_logger(__name__)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def save_artifact(obj, path: str) -> None:
    """Serialize any sklearn/XGBoost object to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    logging.getLogger(__name__).info(f"Artifact saved → {path}")


def load_artifact(path: str):
    """Deserialize a joblib artifact from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No artifact found at: {path}")
    obj = joblib.load(path)
    logging.getLogger(__name__).info(f"Artifact loaded ← {path}")
    return obj
