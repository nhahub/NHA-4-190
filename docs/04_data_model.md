# Data Model

This project does **not** use a relational database in the default deployment. Operational data is stored in **CSV files**, **YAML configuration**, and **binary model artifacts**. The following **logical model** satisfies documentation expectations for entities, keys, and relationships.

## 1. Entity-relationship view (logical)

```mermaid
erDiagram
  ASSET ||--o{ PREDICTION_LOG : "identified_by"
  PREDICTION_LOG ||--o| FEEDBACK_LOG : "request_id"
  TRAINING_RUN ||--|| MODEL_ARTIFACT : "produces"
  MODEL_ARTIFACT ||--o| THRESHOLD_META : "uses"
  CONFIG ||--o{ TRAINING_RUN : "parameterizes"

  ASSET {
    string asset_id PK
  }

  PREDICTION_LOG {
    string timestamp
    string request_id PK
    string asset_id FK
    float failure_probability
    int predicted_failure
    string feature_columns "wide feature vector"
  }

  FEEDBACK_LOG {
    string timestamp
    string request_id PK_FK
    int actual_failure
  }

  TRAINING_RUN {
    string run_id PK
    datetime started_at
    string config_hash
  }

  MODEL_ARTIFACT {
    string path PK
    string estimator_type
    datetime trained_at
  }

  THRESHOLD_META {
    string path PK
    float decision_threshold
  }

  CONFIG {
    string path PK
    string version_note
  }
```

*Note: `PREDICTION_LOG` stores engineered features as additional columns for drift (PSI) and debugging; exact columns match `feature_names_in_` plus metadata.*

## 2. Physical / storage mapping

| Logical entity | Physical location | Format |
|----------------|-------------------|--------|
| Raw training data | `data/raw/ai4i2020.csv` | CSV (user-supplied) |
| Configuration | `config/config.yaml` | YAML |
| Trained model | `models/best_model.joblib` (configurable) | joblib |
| Decision threshold | e.g. `models/random_forest_(optimized)_threshold.json` | JSON |
| Prediction log | `outputs/monitoring/predictions_log.csv` | CSV |
| Feedback log | `outputs/monitoring/feedback_log.csv` | CSV |
| Monitoring report | `outputs/monitoring/latest_monitoring_report.json` | JSON |
| Evaluation plots | `outputs/plots/` | PNG |

Paths are controlled by **`config/config.yaml`** under `deployment` and `monitoring`.

## 3. Keys and integrity rules

- **`request_id`:** UUID per prediction; **primary key** for joining feedback to predictions.
- **`asset_id`:** Identifies equipment for rolling-window state in `RollingFeatureStore`; default `"default"` if omitted.
- **No foreign-key enforcement** at storage layer (CSV); application code must pass consistent `request_id` values into feedback.

## 4. Normalization note

Wide feature columns in `predictions_log.csv` are **denormalized** on purpose for:

- Per-feature PSI drift without joins  
- Simple append-only logging from the API  

A warehouse implementation would split **fact** (prediction) and **dimension** (feature vectors) — out of scope for this repo.
