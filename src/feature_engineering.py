"""
feature_engineering.py
-----------------------
Milestone 2: Physics-informed + time-series feature engineering.
PCA is intentionally NOT done here — it belongs only in the
visualization notebook (unsupervised, 2D, for exploration only).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-driven features grounded in thermodynamics
    and mechanical engineering.

      - Temp_Diff : Process_Temp − Air_Temp → thermal stress indicator
      - Power     : Torque × Rotational_Speed → mechanical load proxy
    """
    df = df.copy()
    df["Temp_Diff"] = df["Process_Temp"] - df["Air_Temp"]
    df["Power"] = df["Torque"] * df["Rotational_Speed"]
    logger.info("Physics features added: Temp_Diff, Power")
    return df


def add_time_series_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add degradation-tracking features over a rolling window.

      - Temp_Rate_of_Change   : % change in Process_Temp per step
      - Torque_Rate_of_Change : % change in Torque per step
      - Rolling_Avg_Temp      : smoothed temperature trend
      - Rolling_Std_Torque    : torque instability (variance proxy)

    Parameters
    ----------
    df     : DataFrame with physics features already added
    window : Rolling window size (default 5 from config)
    """
    df = df.copy()
    df["Temp_Rate_of_Change"] = df["Process_Temp"].pct_change().fillna(0)
    df["Torque_Rate_of_Change"] = df["Torque"].pct_change().fillna(0)
    df["Rolling_Avg_Temp"] = (
        df["Process_Temp"].rolling(window=window, min_periods=1).mean()
    )
    df["Rolling_Std_Torque"] = (
        df["Torque"].rolling(window=window, min_periods=1).std().fillna(0)
    )
    logger.info(
        f"Time-series features added (window={window}): "
        "Temp_Rate_of_Change, Torque_Rate_of_Change, "
        "Rolling_Avg_Temp, Rolling_Std_Torque"
    )
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'Type' column (L/M/H machine type).
    drop_first=True avoids the dummy variable trap.
    """
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)
    logger.info(f"After encoding, columns: {list(df.columns)}")
    return df


def fit_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame]:
    """
    Fit a StandardScaler on training data and return (scaler, scaled_X_train).
    The scaler must be saved and reused on X_test / live data — never refit.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    logger.info("Scaler fitted on training data.")
    return scaler, X_train_scaled


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Transform data using a previously fitted scaler (no refit)."""
    return pd.DataFrame(scaler.transform(X), columns=X.columns)


def run_feature_engineering(
    df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """
    Full Milestone 2 pipeline (no PCA — that's visualization only).
    Steps: physics → time-series → encode categorical.
    """
    window = config["features"]["rolling_window"]
    df = add_physics_features(df)
    df = add_time_series_features(df, window=window)
    df = encode_categorical(df)
    logger.info(f"Final engineered shape: {df.shape}")
    return df


if __name__ == "__main__":
    from src.data_preprocessing import run_preprocessing
    from src.utils import load_config

    config = load_config()
    df_clean = run_preprocessing()
    df_engineered = run_feature_engineering(df_clean, config)
    print(df_engineered.head())
    print(df_engineered.columns.tolist())
