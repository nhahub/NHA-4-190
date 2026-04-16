"""
serving.py
----------
Milestone 4 helpers for online inference, logging, and drift monitoring.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import csv
import json
import os
from typing import Any

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RollingFeatureStore:
    """In-memory rolling history by asset_id for online feature generation."""

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._temp_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._torque_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def compute(self, asset_id: str, process_temp: float, torque: float) -> dict[str, float]:
        temp_q = self._temp_history[asset_id]
        torque_q = self._torque_history[asset_id]

        prev_temp = temp_q[-1] if temp_q else None
        prev_torque = torque_q[-1] if torque_q else None

        temp_roc = ((process_temp - prev_temp) / prev_temp) if prev_temp not in (None, 0) else 0.0
        torque_roc = ((torque - prev_torque) / prev_torque) if prev_torque not in (None, 0) else 0.0

        temp_q.append(float(process_temp))
        torque_q.append(float(torque))

        rolling_avg_temp = float(np.mean(temp_q))
        rolling_std_torque = float(np.std(torque_q))

        return {
            "Temp_Rate_of_Change": float(temp_roc),
            "Torque_Rate_of_Change": float(torque_roc),
            "Rolling_Avg_Temp": rolling_avg_temp,
            "Rolling_Std_Torque": rolling_std_torque,
        }


def build_online_features(event: dict[str, Any], feature_store: RollingFeatureStore) -> dict[str, float]:
    """Build one online feature row matching the training feature schema."""
    asset_id = str(event.get("asset_id", "default"))

    air_temp = float(event["Air_Temp"])
    process_temp = float(event["Process_Temp"])
    rotational_speed = float(event["Rotational_Speed"])
    torque = float(event["Torque"])
    tool_wear = float(event["Tool_Wear"])

    rolling_features = feature_store.compute(asset_id, process_temp, torque)

    machine_type = str(event["Type"]).upper().strip()

    features = {
        "Air_Temp": air_temp,
        "Process_Temp": process_temp,
        "Rotational_Speed": rotational_speed,
        "Torque": torque,
        "Tool_Wear": tool_wear,
        "TWF": int(event.get("TWF", 0)),
        "HDF": int(event.get("HDF", 0)),
        "PWF": int(event.get("PWF", 0)),
        "OSF": int(event.get("OSF", 0)),
        "RNF": int(event.get("RNF", 0)),
        "Temp_Diff": process_temp - air_temp,
        "Power": torque * rotational_speed,
        **rolling_features,
        "Type_L": 1 if machine_type == "L" else 0,
        "Type_M": 1 if machine_type == "M" else 0,
    }

    return features


def align_to_model_features(row: dict[str, float], expected_features: list[str]) -> pd.DataFrame:
    """Align an online feature dict to model expected feature order."""
    aligned = {feature: row.get(feature, 0.0) for feature in expected_features}
    return pd.DataFrame([aligned], columns=expected_features)


def load_threshold(threshold_path: str, fallback: float = 0.5) -> float:
    if not os.path.exists(threshold_path):
        return fallback

    with open(threshold_path, "r") as f:
        payload = json.load(f)
    return float(payload.get("threshold", fallback))


def append_csv_row(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for one feature."""
    if reference.nunique() <= 1 or current.nunique() <= 1:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    cut_points = np.unique(np.quantile(reference, quantiles))
    if len(cut_points) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=cut_points)
    cur_counts, _ = np.histogram(current, bins=cut_points)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-8, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-8, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)
