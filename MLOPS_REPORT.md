# Milestone 4 — MLOps, Deployment, and Continuous Monitoring

## Deployment Pipeline

- Trained model artifact loaded from [models/best_model.joblib](models/best_model.joblib)
- Decision threshold loaded from [models/random_forest_(optimized)_threshold.json](models/random_forest_(optimized)_threshold.json)
- Real-time inference service exposed through [app.py](app.py)
- API served with FastAPI + Uvicorn

## Online Inference Flow

1. IoT sensor payload arrives at `POST /predict`
2. Online features are generated (physics + rolling + rate-of-change)
3. Features are aligned to `model.feature_names_in_`
4. Probability and binary decision are returned
5. Prediction event is logged to [outputs/monitoring/predictions_log.csv](outputs/monitoring/predictions_log.csv)

## Monitoring and Drift

- Feedback labels are accepted by `POST /feedback`
- Monitoring job [monitoring/monitor.py](monitoring/monitor.py) computes:
  - PSI drift between reference and live distributions
  - Precision / Recall / F1 / ROC-AUC on labeled live samples
- Monitoring report is written to [outputs/monitoring/latest_monitoring_report.json](outputs/monitoring/latest_monitoring_report.json)

## Retraining Trigger

- Trigger conditions are configured in [config/config.yaml](config/config.yaml):
  - `monitoring.psi_threshold`
  - `monitoring.recall_threshold`
  - `monitoring.min_labeled_samples`
- Optional automatic retraining:
  - `python monitoring/monitor.py --retrain --skip-optimization`

## Versioning and Reproducibility

- Configuration and thresholds are centralized in [config/config.yaml](config/config.yaml)
- Model artifacts are persisted in [models](models)
- End-to-end training pipeline remains reproducible via [main.py](main.py)
