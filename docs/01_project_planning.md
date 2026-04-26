# Project planning and management

## Proposal summary

**NHA-4-190 — AI-powered predictive maintenance for industrial equipment**

The project delivers a reproducible ML pipeline on the AI4I 2020 dataset, a deployed FastAPI inference service that matches offline feature generation, and monitoring (PSI drift, aggregate metrics, optional retrain) driven by `config/config.yaml`.

**In scope:** binary failure prediction; endpoints `/health`, `/predict`, `/feedback`, `/metrics`, `/drift`; artifacts as joblib models, threshold JSON, CSV logs, and YAML configuration.

**Out of scope for this release:** multi-class failure typing, remaining useful life (RUL), deep learning models, managed cloud deployment, authentication layer, and a full browser-based operations UI. Those items are noted under [Future work](03_system_design.md#future-work) in the design document.

## Milestones

| Phase | Output |
|--------|--------|
| M1 | Preprocessing in `src/data_preprocessing.py`; validated inputs |
| M2 | Feature engineering in `src/feature_engineering.py`; exploratory analysis in `notebooks/` |
| M3 | Random Forest and XGBoost training and evaluation; thresholds; `main.py` orchestration |
| M4 | `app.py`, `src/serving.py`, `monitoring/monitor.py`; prediction and feedback logs; drift checks |

## Schedule

The Gantt below is illustrative; replace dates with your programme calendar.

```mermaid
gantt
    title NHA-4-190 — schedule (indicative)
    dateFormat  YYYY-MM-DD
    section Setup
    Proposal and environment           :m0, 2026-01-15, 14d
    section Modelling
    Preprocessing and features M1 M2   :m1, after m0, 21d
    Training and evaluation M3         :m2, after m1, 21d
    section Operations
    API and serving M4                 :m3, after m2, 14d
    Monitoring and documentation       :m4, after m3, 14d
```

## Roles and responsibilities

| Area | Responsibility |
|------|----------------|
| Data and features | Preprocessing, feature engineering, run reproducibility |
| Modelling | Training, optional grid search, metrics, threshold selection |
| Deployment | FastAPI service, rolling feature store, schema alignment with the trained model |
| Quality and MLOps | Monitoring job, drift checks, test documentation, CI |

Update the table with team member names as appropriate.

## Risks and mitigation

| Risk | Mitigation |
|------|------------|
| Strong class imbalance (~3.4% positives) | SMOTE on the training split only; Fβ threshold tuning with β = 2 |
| Temporal leakage in evaluation | `shuffle: false` in config; resampling after the train/test split |
| Training–serving skew | Shared YAML config; `RollingFeatureStore` and column alignment in `src/serving.py` |
| Input or concept drift | PSI against the training reference; thresholds in config; optional pipeline re-run |
| Missing or wrong data path | Document `data/raw/ai4i2020.csv`; README lists prerequisites |

## Key performance indicators

| Measure | Target | Notes |
|---------|--------|--------|
| Recall on failures | ≥ 0.80 where applicable | Held-out evaluation or feedback-backed monitoring |
| Service readiness | `GET /health` returns stable metadata | Model name, threshold, feature count |
| Drift signal | Align with `monitoring.psi_threshold` | Exposed on `GET /drift` |
| Reproducibility | Deterministic training from fixed config and data | Seeds and paths in `config/config.yaml` |

## Version control

Development uses `main` as the integration branch. Short-lived branches follow `feature/<topic>` and merge back through pull requests when possible. Commit messages should state the change in imperative form (for example: `Add drift endpoint`, `Fix log path in config`).
