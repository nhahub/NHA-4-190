# AI-Powered Predictive Maintenance for Industrial Equipment

Predicts equipment failures before they occur using IoT sensor data and machine learning.

## Documentation (course / design)

Structured deliverables live in **[`docs/`](docs/README.md)** — planning, requirements, system design (with diagrams), data model, testing, user manual, and a template for instructor feedback.

## System requirements

- **Python:** 3.10+ recommended (CI uses 3.11).
- **OS:** Windows, macOS, or Linux.
- **Hardware:** Sufficient RAM for pandas/sklearn (GridSearch is heavier; use `python main.py --skip-optimization` on small machines).
- **Dataset:** AI4I 2020 CSV placed at `data/raw/ai4i2020.csv` before training.

## Dataset
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
Place the CSV at `data/raw/ai4i2020.csv`.

## Project Structure

```
NHA-4-190/
├── config/
│   └── config.yaml          # All hyperparameters and paths (single source of truth)
├── data/
│   └── raw/                 # Place ai4i2020.csv here
├── docs/                    # Course documentation (planning, design, testing, user manual)
├── models/                  # Saved .joblib model artifacts + threshold JSON
├── monitoring/
│   └── monitor.py           # PSI / metrics / optional --retrain
├── notebooks/
│   └── visualization.ipynb  # EDA + degradation plots + PCA (Colab-friendly)
├── outputs/
│   ├── plots/               # Confusion matrix PNGs
│   └── monitoring/          # Prediction & feedback logs, monitoring report
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── serving.py           # M4: online features, logging, PSI helpers
│   └── utils.py
├── app.py                   # FastAPI service (Milestone 4)
├── main.py                  # Pipeline entry point
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/nhahub/NHA-4-190.git
cd NHA-4-190
pip install -r requirements.txt

# 2. Add dataset
cp /path/to/ai4i2020.csv data/raw/

# 3. Run the full pipeline
python main.py

# 4. Skip GridSearch for a fast debug run
python main.py --skip-optimization
```

## Milestone 4 (MLOps, Deployment, Monitoring)

### 1) Start real-time API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2) API endpoints

- `GET /health` → service + model status
- `POST /predict` → real-time failure prediction
- `POST /feedback` → send true outcome labels for online evaluation
- `GET /metrics` → prediction volume + alert rate
- `GET /drift` → PSI-based drift check

### 3) Run monitoring job

```bash
python monitoring/monitor.py
```

Optional auto-retrain when thresholds are breached:

```bash
python monitoring/monitor.py --retrain --skip-optimization
```

## Key Design Decisions

- **No shuffle in train/test split** — temporal order is preserved for time-series integrity
- **SMOTE on training data only** — prevents data leakage from the test set
- **PCA excluded from production pipeline** — PCA is for visualization/exploration; the models train on all engineered features
- **Central config** — all hyperparameters live in `config/config.yaml`, never hardcoded
- **Joblib serialization** — trained models saved to `/models` for Milestone 4 deployment

## Features Engineered

| Feature | Type | Rationale |
|---------|------|-----------|
| `Temp_Diff` | Physics | Process − Air temp; thermal stress indicator |
| `Power` | Physics | Torque × RPM; mechanical load proxy |
| `Temp_Rate_of_Change` | Time-series | Rate of thermal change per step |
| `Torque_Rate_of_Change` | Time-series | Rate of torque change per step |
| `Rolling_Avg_Temp` | Time-series | Smoothed temperature trend |
| `Rolling_Std_Torque` | Time-series | Torque instability measure |
