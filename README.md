# AI-Powered Predictive Maintenance for Industrial Equipment

Predicts equipment failures before they occur using IoT sensor data and machine learning.

## Dataset
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
Place the CSV at `data/raw/ai4i2020.csv`.

## Project Structure

```
predictive_maintenance/
├── config/
│   └── config.yaml          # All hyperparameters and paths (single source of truth)
├── data/
│   └── raw/                 # Place ai4i2020.csv here
├── models/                  # Saved .joblib model artifacts
├── notebooks/
│   └── visualization.ipynb  # EDA + degradation plots + PCA (Colab-friendly)
├── outputs/
│   └── plots/               # Confusion matrix PNGs saved here
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── main.py                  # Pipeline entry point
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/nhahub/NHA-4-190.git
cd predictive_maintenance
pip install -r requirements.txt

# 2. Add dataset
cp /path/to/ai4i2020.csv data/raw/

# 3. Run the full pipeline
python main.py

# 4. Skip GridSearch for a fast debug run
python main.py --skip-optimization
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
