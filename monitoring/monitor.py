"""
monitor.py
----------
Milestone 4 monitoring job.
Computes drift (PSI), online performance from feedback labels, and optional retraining trigger.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.serving import calculate_psi, utc_now_iso
from src.utils import get_logger, load_config

logger = get_logger("monitor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run predictive maintenance monitoring checks.")
    parser.add_argument("--retrain", action="store_true", help="Retrain automatically if thresholds are breached.")
    parser.add_argument("--skip-optimization", action="store_true", help="Use fast retraining mode.")
    return parser


def compute_drift(config: dict, current: pd.DataFrame) -> tuple[float, dict[str, float]]:
    reference_raw = run_preprocessing("config/config.yaml")
    reference_eng = run_feature_engineering(reference_raw, config)
    reference_X = reference_eng.drop(columns=["Machine_Failure"])

    shared_cols = [c for c in current.columns if c in reference_X.columns]
    psi_by_feature = {col: calculate_psi(reference_X[col], current[col]) for col in shared_cols}

    avg_psi = float(sum(psi_by_feature.values()) / max(len(psi_by_feature), 1))
    return avg_psi, psi_by_feature


def compute_online_performance(pred_log: pd.DataFrame, feedback_log: pd.DataFrame) -> dict:
    merged = pred_log.merge(feedback_log[["request_id", "actual_failure"]], on="request_id", how="inner")

    if merged.empty:
        return {
            "labeled_samples": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
        }

    y_true = merged["actual_failure"].astype(int)
    y_pred = merged["predicted_failure"].astype(int)
    y_prob = merged["failure_probability"].astype(float)

    metrics = {
        "labeled_samples": int(len(merged)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }

    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    return metrics


def maybe_retrain(breached: bool, skip_optimization: bool) -> bool:
    if not breached:
        return False

    cmd = ["python", "main.py"]
    if skip_optimization:
        cmd.append("--skip-optimization")

    logger.warning("Threshold breached. Triggering retraining...")
    subprocess.run(cmd, check=True)
    return True


def main() -> None:
    args = build_parser().parse_args()
    config = load_config("config/config.yaml")
    deployment_cfg = config["deployment"]
    monitoring_cfg = config["monitoring"]

    pred_path = deployment_cfg["log_path"]
    feedback_path = deployment_cfg["feedback_log_path"]

    if not pd.io.common.file_exists(pred_path):
        logger.warning("No prediction logs found yet.")
        return

    pred_log = pd.read_csv(pred_path)

    avg_psi, psi_by_feature = compute_drift(config, pred_log)
    drift_detected = avg_psi >= float(monitoring_cfg["psi_threshold"])

    if pd.io.common.file_exists(feedback_path):
        feedback_log = pd.read_csv(feedback_path)
    else:
        feedback_log = pd.DataFrame(columns=["request_id", "actual_failure"])

    online_metrics = compute_online_performance(pred_log, feedback_log)

    recall_threshold = float(monitoring_cfg["recall_threshold"])
    min_labeled = int(monitoring_cfg["min_labeled_samples"])

    recall_breach = (
        online_metrics["recall"] is not None
        and online_metrics["labeled_samples"] >= min_labeled
        and online_metrics["recall"] < recall_threshold
    )

    breached = bool(drift_detected or recall_breach)

    retrained = False
    if args.retrain:
        retrained = maybe_retrain(breached, args.skip_optimization)

    report = {
        "timestamp": utc_now_iso(),
        "summary": {
            "drift_detected": drift_detected,
            "avg_psi": avg_psi,
            "psi_threshold": monitoring_cfg["psi_threshold"],
            "recall_breach": recall_breach,
            "recall_threshold": recall_threshold,
            "retrained": retrained,
        },
        "online_metrics": online_metrics,
        "top_psi_features": dict(sorted(psi_by_feature.items(), key=lambda x: x[1], reverse=True)[:10]),
    }

    report_path = monitoring_cfg["report_path"]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Monitoring report saved → %s", report_path)


if __name__ == "__main__":
    main()
