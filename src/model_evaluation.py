"""
model_evaluation.py
-------------------
Milestone 3: Evaluate trained models and save results.
Produces classification reports, ROC-AUC scores, and saves
confusion matrix plots to disk (no plt.show() for headless runs).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

from src.utils import get_logger
from sklearn.metrics import precision_recall_curve
import json

logger = get_logger(__name__)

PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save_plot: bool = True,
) -> dict:
    """
    Evaluate a single classifier and optionally save its confusion matrix.

    Parameters
    ----------
    model      : Fitted sklearn-compatible classifier
    X_test     : Test features
    y_test     : True labels
    model_name : Display name for logging / plot titles
    save_plot  : Whether to save the confusion matrix PNG

    Returns
    -------
    dict with precision, recall, f1, roc_auc, and the full report string
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Fixed: was named roc_auc_auc

    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    logger.info(f"\n{'='*50}\n{model_name}\n{'='*50}")
    logger.info(f"\n{report}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")

    if save_plot:
        _save_confusion_matrix(
            confusion_matrix(y_test, y_pred),
            model_name=model_name,
        )

    return {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "report": report,
    }


def _save_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """Save a confusion matrix heatmap as a PNG file."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {path}")

def find_best_threshold(model, X_test, y_test, model_name: str, beta: int = 2) -> float:
    """
    Find the probability threshold that maximizes F-beta score.
    beta=2 weights recall 2x more than precision — correct for
    predictive maintenance where missing a failure is far costlier
    than a false alarm.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    f_beta = (
        (1 + beta**2) * (precisions * recalls)
        / (beta**2 * precisions + recalls + 1e-8)
    )
    best_idx = f_beta.argmax()
    best_threshold = float(thresholds[best_idx])

    logger.info(
        f"{model_name} optimal threshold: {best_threshold:.3f} "
        f"→ Precision: {precisions[best_idx]:.3f}, "
        f"Recall: {recalls[best_idx]:.3f}"
    )

    # Save threshold to disk so the deployment API can load it
    os.makedirs("models", exist_ok=True)
    threshold_path = f"models/{model_name.lower().replace(' ', '_')}_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"model": model_name, "threshold": best_threshold, "beta": beta}, f)
    logger.info(f"Threshold saved → {threshold_path}")

    return best_threshold

def evaluate_all_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    """
    Evaluate every model in the dict and return a summary list.

    Parameters
    ----------
    models : {"display_name": fitted_model, ...}

    Returns
    -------
    List of result dicts from evaluate_model()
    """
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, model_name=name)
        # ← ADD THIS: find and save threshold for each model
        threshold = find_best_threshold(model, X_test, y_test, model_name=name)
        result["best_threshold"] = threshold  # ← ADD THIS too

        results.append(result)

    # Print leaderboard
    logger.info("\n--- Model Leaderboard (by ROC-AUC) ---")
    for r in sorted(results, key=lambda x: x["roc_auc"], reverse=True):
        logger.info(
            f"  {r['model_name']}: ROC-AUC = {r['roc_auc']:.4f} "
            f"| Best Threshold = {r['best_threshold']:.3f}"  # ← shows in leaderboard now
        )

    return results


if __name__ == "__main__":
    from src.utils import load_config, load_artifact
    from src.data_preprocessing import run_preprocessing
    from src.feature_engineering import run_feature_engineering
    from src.model_training import split_data

    config = load_config()
    df_clean = run_preprocessing()
    df_eng = run_feature_engineering(df_clean, config)
    _, _, X_test, _, _, y_test = split_data(df_eng, config)

    artifact_cfg = config["artifacts"]
    model_dir = artifact_cfg["model_dir"]

    models = {
        "Random Forest (Baseline)":  load_artifact(f"{model_dir}/{artifact_cfg['rf_model_name']}"),
        "Random Forest (Optimized)": load_artifact(f"{model_dir}/{artifact_cfg['best_model_name']}"),
        "XGBoost (Baseline)":        load_artifact(f"{model_dir}/{artifact_cfg['xgb_model_name']}"),
    }

    evaluate_all_models(models, X_test, y_test)
