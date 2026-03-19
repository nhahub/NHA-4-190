"""
model_training.py
-----------------
Milestone 3: Data splitting, SMOTE, model training, and hyperparameter
tuning. All trained models are returned for evaluation and serialization.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils import get_logger, load_config, save_artifact

logger = get_logger(__name__)


# ── Type alias for readability ──────────────────────────────────────────────
SplitData = tuple[
    pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series,
    pd.Series, pd.Series,
]


def split_data(df: pd.DataFrame, config: dict) -> SplitData:
    """
    Time-series aware split: shuffle=False preserves temporal order.
    Returns both the raw train split and the SMOTE-balanced version.

    Returns
    -------
    X_train, X_train_smote, X_test, y_train, y_train_smote, y_test
    """
    X = df.drop(columns=["Machine_Failure"])
    y = df["Machine_Failure"]

    test_size = config["data"]["test_size"]
    shuffle = config["data"]["shuffle"]  # Must be False for time-series

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )
    logger.info(
        f"Split → Train: {len(X_train)}, Test: {len(X_test)} | "
        f"Failures in train: {y_train.sum()}"
    )

    # Apply SMOTE ONLY to training data
    smote = SMOTE(random_state=config["smote"]["random_state"])
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    logger.info(
        f"After SMOTE → Train: {len(X_train_smote)} | "
        f"Failures: {y_train_smote.sum()}"
    )

    return X_train, X_train_smote, X_test, y_train, y_train_smote, y_test


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict
) -> RandomForestClassifier:
    """Train a baseline Random Forest classifier."""
    rf_cfg = config["models"]["random_forest"]
    rf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        random_state=rf_cfg["random_state"],
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return rf


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, config: dict
) -> XGBClassifier:
    """Train a baseline XGBoost classifier."""
    xgb_cfg = config["models"]["xgboost"]
    xgb = XGBClassifier(
        eval_metric=xgb_cfg["eval_metric"],
        random_state=xgb_cfg["random_state"],
    )
    xgb.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return xgb


def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> XGBClassifier:
    """
    Grid search over XGBoost hyperparameters.
    Scoring is 'recall' — catching failures is the #1 priority.

    Returns
    -------
    Best XGBClassifier found by GridSearchCV
    """
    gs_cfg = config["models"]["xgb_grid_search"]
    xgb_base = XGBClassifier(
        eval_metric=config["models"]["xgboost"]["eval_metric"],
        random_state=config["models"]["xgboost"]["random_state"],
    )

    param_grid = {
        "max_depth": gs_cfg["max_depth"],
        "learning_rate": gs_cfg["learning_rate"],
        "n_estimators": gs_cfg["n_estimators"],
    }

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring=gs_cfg["scoring"],
        cv=gs_cfg["cv"],
        verbose=1,
        n_jobs=gs_cfg["n_jobs"],
    )

    logger.info("Starting XGBoost Grid Search (this may take a few minutes)…")
    grid_search.fit(X_train, y_train)

    logger.info(f"Best params: {grid_search.best_params_}")
    return grid_search.best_estimator_

def optimize_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> RandomForestClassifier:
    """GridSearch on RF — optimized for recall."""
    rf_base = RandomForestClassifier(random_state=config["models"]["random_forest"]["random_state"])

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring="recall",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )

    logger.info("Starting Random Forest Grid Search...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best RF params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def run_training_pipeline(
    df_engineered: pd.DataFrame,
    config: dict,
    save_models: bool = True,
) -> dict:
    """
    Full Milestone 3 pipeline.

    Steps
    -----
    1. Split data (time-series aware)
    2. SMOTE balancing
    3. Train Random Forest
    4. Train baseline XGBoost
    5. Optimize XGBoost with GridSearch
    6. Save all models to disk (optional)

    Returns
    -------
    dict with keys: rf, xgb_base, xgb_best, X_test, y_test
    """
    (
        X_train, X_train_smote, X_test,
        y_train, y_train_smote, y_test,
    ) = split_data(df_engineered, config)

    rf_model = train_random_forest(X_train_smote, y_train_smote, config)
    rf_best_model = optimize_random_forest(X_train_smote, y_train_smote, config)  # ← add
    xgb_base_model = train_xgboost(X_train_smote, y_train_smote, config)
    xgb_best_model = optimize_xgboost(X_train_smote, y_train_smote, config)

    if save_models:
        artifact_cfg = config["artifacts"]
        model_dir = artifact_cfg["model_dir"]
        save_artifact(rf_model,       f"{model_dir}/{artifact_cfg['rf_model_name']}")
        save_artifact(rf_best_model,  f"{model_dir}/{artifact_cfg['best_model_name']}")  # ← rf_best_model now
        save_artifact(xgb_base_model, f"{model_dir}/{artifact_cfg['xgb_model_name']}")

    return {
        "rf": rf_model,
        "rf_best": rf_best_model,   # ← add this line
        "xgb_base": xgb_base_model,
        "xgb_best": xgb_best_model,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    from src.data_preprocessing import run_preprocessing
    from src.feature_engineering import run_feature_engineering
    from src.utils import load_config

    config = load_config()
    df_clean = run_preprocessing()
    df_eng = run_feature_engineering(df_clean, config)
    results = run_training_pipeline(df_eng, config)
    print("Models trained and saved.")
