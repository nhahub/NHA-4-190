# System analysis and design

## Problem and goals

Unplanned failures drive downtime and cost; calendar-based maintenance wastes effort. The system estimates failure risk from sensor streams so maintenance can be triggered on condition rather than only on schedule or after breakdown.

Goals: ingest readings compatible with AI4I 2020, apply the same feature logic in batch and online settings, train and select a binary classifier under imbalance, expose inference over HTTP, and support basic production checks (logging, drift, optional retrain).

## Architecture

Training is batch-oriented (`main.py`, modules under `src/`). Inference is a FastAPI application (`app.py`) that reuses preprocessing and feature logic via `src/serving.py`. Both paths read `config/config.yaml` and the same joblib model and threshold files.

```mermaid
flowchart LR
  subgraph offline["Batch training"]
    A[Raw CSV] --> B[Preprocess]
    B --> C[Feature engineering]
    C --> D[Train and evaluate]
    D --> E[Model and threshold files]
  end
  subgraph online["Online inference"]
    F[Client] --> G[FastAPI]
    G --> H[Rolling feature store]
    H --> I[Scoring]
    I --> J[CSV logs]
  end
  E --> I
  C -. aligned logic .-> H
```

The structure is a small service-oriented layout: HTTP layer, scoring and feature helpers, and file-backed persistence.

## Use cases (diagram)

```mermaid
flowchart TB
  Client[Client system]
  Ops[Operator or engineer]
  Client --> UC1[Predict failure]
  Client --> UC4[View metrics]
  Client --> UC5[Check drift]
  Ops --> UC2[Submit feedback]
  Ops --> UC3[Health check]
```

## Components

```mermaid
flowchart TB
  subgraph api["API"]
    app[app.py]
  end
  subgraph core["ML and features"]
    prep[data_preprocessing]
    fe[feature_engineering]
    serve[serving]
    train[model_training]
    eval[model_evaluation]
  end
  subgraph store["Files"]
    cfg[config.yaml]
    mdl[best_model.joblib]
    thr[threshold JSON]
    log[predictions_log.csv]
    fb[feedback_log.csv]
  end
  app --> serve
  app --> prep
  serve --> fe
  train --> mdl
  train --> thr
  app --> log
  app --> fb
  train --> cfg
  serve --> cfg
```

## Deployment (logical)

```mermaid
flowchart TB
  subgraph machine["Host"]
    uv[Uvicorn]
    proc[FastAPI process]
    job[monitor.py]
  end
  subgraph fs["Filesystem"]
    cfg[config/]
    models[models/]
    out[outputs/monitoring/]
  end
  Client[HTTP clients] --> uv
  uv --> proc
  proc --> fs
  job --> fs
  job --> cfg
```

A hardened deployment would add a reverse proxy, TLS, process supervision, and remote artifact storage.

## Sequence: prediction

```mermaid
sequenceDiagram
  participant C as Client
  participant A as FastAPI
  participant S as Serving layer
  participant M as Model
  participant L as Prediction log
  C->>A: POST /predict
  A->>S: Build and align features
  S->>M: predict_proba
  M-->>S: Probability
  A->>L: Append row
  A-->>C: JSON response
```

## Activity: monitoring

```mermaid
flowchart TD
  Start([monitor.py]) --> Read[Load logs and thresholds]
  Read --> Check{Breach?}
  Check -->|No| Report[Write report]
  Check -->|Yes| Retrain{Retrain flag}
  Retrain -->|No| Report
  Retrain -->|Yes| Run[Run main.py]
  Run --> Report
  Report --> End([Exit])
```

## Data flow (context)

```mermaid
flowchart LR
  Src[Sensor or client] --> Sys[NHA-4-190]
  Sys --> Out[Predictions]
  Op[Operator] --> Sys
```

Level 1: raw file → preprocess → features → score → response; logs feed metrics and drift.

## Request state (simplified)

```mermaid
stateDiagram-v2
  [*] --> Received
  Received --> Built: Valid payload
  Built --> Scored: Inference
  Scored --> Logged: Persist
  Logged --> [*]
  Received --> Failed: Error
  Failed --> [*]
```

## API payload models (Pydantic)

```mermaid
classDiagram
  class SensorEvent {
    +asset_id
    +Type
    +Air_Temp
    +Process_Temp
    +Rotational_Speed
    +Torque
    +Tool_Wear
    +TWF
    +HDF
    +PWF
    +OSF
    +RNF
  }
  class FeedbackEvent {
    +request_id
    +actual_failure
  }
```

Estimator internals follow scikit-learn conventions; the design emphasises pipelines and explicit feature names.

## Technology

| Layer | Choice |
|--------|--------|
| Runtime | Python 3.10+ |
| ML | scikit-learn, XGBoost, imbalanced-learn |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Config | PyYAML |
| Model storage | joblib |

## REST surface

Schemas are defined in `app.py`. With the server running, OpenAPI is available at `/docs`.

| Method | Path | Role |
|--------|------|------|
| GET | `/health` | Liveness and model metadata |
| POST | `/predict` | Score one observation |
| POST | `/feedback` | Attach label to `request_id` |
| GET | `/metrics` | Aggregates from the prediction log |
| GET | `/drift` | PSI summary vs training reference |

## Future work

- Sequence models (e.g. LSTM or transformer) for longer histories  
- Per failure-mode classification  
- RUL estimation  
- On-device or edge inference  
- Operator dashboard  
- Controlled experiments on retrained models  
