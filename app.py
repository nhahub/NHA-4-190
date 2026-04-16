"""
app.py
------
Milestone 4 FastAPI service for real-time predictive maintenance inference.
"""

from __future__ import annotations

import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.serving import (
    RollingFeatureStore,
    align_to_model_features,
    append_csv_row,
    build_online_features,
    calculate_psi,
    load_threshold,
    utc_now_iso,
)
from src.utils import get_logger, load_artifact, load_config

logger = get_logger("api")

app = FastAPI(
    title="Predictive Maintenance API",
    version="1.0.0",
    description="Real-time failure prediction + monitoring for Milestone 4.",
)

config = load_config("config/config.yaml")
deployment_cfg = config["deployment"]
monitoring_cfg = config["monitoring"]

model = load_artifact(deployment_cfg["model_path"])
threshold = load_threshold(deployment_cfg["threshold_path"], fallback=0.5)
model_name = type(model).__name__

feature_store = RollingFeatureStore(window=config["features"]["rolling_window"])

if not hasattr(model, "feature_names_in_"):
    raise RuntimeError("Loaded model does not expose feature_names_in_.")

expected_features = list(model.feature_names_in_)


class SensorEvent(BaseModel):
    asset_id: str = Field(default="default")
    Type: str = Field(pattern="^(L|M|H)$")
    Air_Temp: float
    Process_Temp: float
    Rotational_Speed: float
    Torque: float
    Tool_Wear: float
    TWF: int = 0
    HDF: int = 0
    PWF: int = 0
    OSF: int = 0
    RNF: int = 0


class FeedbackEvent(BaseModel):
    request_id: str
    actual_failure: int = Field(ge=0, le=1)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": model_name,
        "threshold": threshold,
        "expected_features": len(expected_features),
    }


@app.post("/predict")
def predict(payload: SensorEvent) -> dict:
    try:
        online_features = build_online_features(payload.model_dump(), feature_store)
        X = align_to_model_features(online_features, expected_features)

        failure_probability = float(model.predict_proba(X)[:, 1][0])
        will_fail = int(failure_probability >= threshold)

        request_id = str(uuid.uuid4())
        prediction_log = {
            "timestamp": utc_now_iso(),
            "request_id": request_id,
            "asset_id": payload.asset_id,
            "failure_probability": failure_probability,
            "predicted_failure": will_fail,
            **online_features,
        }
        append_csv_row(deployment_cfg["log_path"], prediction_log)

        return {
            "request_id": request_id,
            "model": model_name,
            "threshold": threshold,
            "failure_probability": failure_probability,
            "predicted_failure": will_fail,
        }
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback")
def feedback(payload: FeedbackEvent) -> dict:
    row = {
        "timestamp": utc_now_iso(),
        "request_id": payload.request_id,
        "actual_failure": payload.actual_failure,
    }
    append_csv_row(deployment_cfg["feedback_log_path"], row)
    return {"status": "recorded", "request_id": payload.request_id}


@app.get("/metrics")
def metrics() -> dict:
    log_path = deployment_cfg["log_path"]
    if not pd.io.common.file_exists(log_path):
        return {"total_predictions": 0, "predicted_failure_rate": 0.0}

    df = pd.read_csv(log_path)
    total = int(len(df))
    failure_rate = float(df["predicted_failure"].mean()) if total else 0.0

    return {
        "total_predictions": total,
        "predicted_failure_rate": failure_rate,
    }


@app.get("/drift")
def drift() -> dict:
    log_path = deployment_cfg["log_path"]
    if not pd.io.common.file_exists(log_path):
        return {"status": "insufficient_data", "reason": "no prediction logs yet"}

    current = pd.read_csv(log_path)
    if len(current) < 100:
        return {"status": "insufficient_data", "reason": "need at least 100 predictions"}

    reference_raw = run_preprocessing("config/config.yaml")
    reference_eng = run_feature_engineering(reference_raw, config)
    reference_X = reference_eng.drop(columns=["Machine_Failure"])

    psi_by_feature = {}
    for col in expected_features:
        if col in current.columns and col in reference_X.columns:
            psi_by_feature[col] = calculate_psi(reference_X[col], current[col])

    if not psi_by_feature:
        return {"status": "insufficient_data", "reason": "no overlapping features"}

    avg_psi = float(sum(psi_by_feature.values()) / len(psi_by_feature))
    drift_detected = avg_psi >= float(monitoring_cfg["psi_threshold"])

    return {
        "status": "ok",
        "avg_psi": avg_psi,
        "psi_threshold": monitoring_cfg["psi_threshold"],
        "drift_detected": drift_detected,
        "top_features": dict(sorted(psi_by_feature.items(), key=lambda x: x[1], reverse=True)[:5]),
    }
