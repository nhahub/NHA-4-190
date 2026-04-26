# System Analysis & Design

## 1. Problem statement & objectives

**Problem:** Reactive maintenance causes downtime and cost; scheduled maintenance is inefficient. Sensor-rich equipment can support **condition-based** decisions if failure risk is estimated accurately and kept stable in production.

**Objectives:** Ingest IoT-style readings, engineer discriminative features, train binary classifiers with imbalance-aware validation, deploy a **real-time API** with **monitoring** aligned to the training pipeline.

## 2. High-level architecture

The system follows a **modular Python** layout: offline **training pipeline** (`main.py` + `src/`) and online **serving** (`app.py` + `src/serving.py`) sharing **YAML config** and **joblib** artifacts.

```mermaid
flowchart LR
  subgraph offline["Offline training"]
    A[Raw CSV] --> B[Preprocess]
    B --> C[Feature engineering]
    C --> D[Train / evaluate]
    D --> E[(Models + thresholds)]
  end
  subgraph online["Online inference"]
    F[HTTP client] --> G[FastAPI app.py]
    G --> H[RollingFeatureStore]
    H --> I[Model predict_proba]
    I --> J[CSV logs]
  end
  E --> I
  C -. feature parity .-> H
```

**Style:** Layered **API + domain + data** (not full MVC); suitable for a small ML service.

## 3. Use case diagram

```mermaid
flowchart TB
  Client[Client / Integration]
  Ops[Operator / Engineer]
  Client --> UC1[UC-1 Predict failure]
  Client --> UC4[UC-4 View metrics]
  Client --> UC5[UC-5 Check drift]
  Ops --> UC2[UC-2 Submit feedback]
  Ops --> UC3[UC-3 Health check]
```

## 4. Component diagram

```mermaid
flowchart TB
  subgraph api["API layer"]
    app[app.py]
  end
  subgraph domain["Domain / ML"]
    prep[data_preprocessing]
    fe[feature_engineering]
    serve[serving]
    train[model_training]
    eval[model_evaluation]
  end
  subgraph data["Artifacts"]
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
  serve --> cfg
  train --> cfg
```

## 5. Deployment diagram (logical)

```mermaid
flowchart TB
  subgraph host["Host machine / VM"]
    uv[Uvicorn worker]
    api[FastAPI process]
    mon[monitor.py job]
  end
  subgraph disk["Local filesystem"]
    cfg[config/]
    models[models/]
    out[outputs/monitoring/]
  end
  Client[HTTP clients] --> uv
  uv --> api
  api --> disk
  mon --> disk
  mon --> cfg
```

*Production would add reverse proxy, TLS, process manager, and remote artifact store.*

## 6. Sequence — prediction (`POST /predict`)

```mermaid
sequenceDiagram
  participant C as Client
  participant A as FastAPI app.py
  participant S as RollingFeatureStore / serving
  participant M as Model joblib
  participant L as predictions_log.csv
  C->>A: POST /predict (sensor payload)
  A->>S: build_online_features + align columns
  S->>M: predict_proba(X)
  M-->>S: probability
  S-->>A: failure_probability, features row
  A->>L: append row + request_id
  A-->>C: JSON response
```

## 7. Activity — monitoring & optional retrain

```mermaid
flowchart TD
  Start([monitor.py]) --> Read[Read logs + config thresholds]
  Read --> PSI{PSI / recall breach?}
  PSI -->|No| Report[Write monitoring report]
  PSI -->|Yes| Retrain{--retrain?}
  Retrain -->|No| Report
  Retrain -->|Yes| Run[Execute main.py pipeline]
  Run --> Report
  Report --> End([Done])
```

## 8. Data flow (context DFD)

**Context (level 0):**

```mermaid
flowchart LR
  Ext[External sensor / client] -->|Sensor readings| SYS[NHA-4-190 system]
  SYS -->|Predictions + alerts| Ext
  Ops2[Operator] -->|Feedback labels| SYS
```

**Level 1 (summary):** Raw file → preprocess → features → model → API response; parallel path: logs → drift/metrics.

## 9. State diagram — prediction request (simplified)

```mermaid
stateDiagram-v2
  [*] --> Received
  Received --> FeaturesBuilt: validate payload
  FeaturesBuilt --> Scored: align + infer
  Scored --> Logged: write CSV
  Logged --> [*]
  Received --> Error: validation / internal error
  Error --> [*]
```

## 10. Class diagram — API models (Pydantic)

```mermaid
classDiagram
  class SensorEvent {
    +str asset_id
    +str Type
    +float Air_Temp
    +float Process_Temp
    +float Rotational_Speed
    +float Torque
    +float Tool_Wear
    +int TWF
    +int HDF
    +int PWF
    +int OSF
    +int RNF
  }
  class FeedbackEvent {
    +str request_id
    +int actual_failure
  }
```

*Full OO design for sklearn estimators is intentionally shallow; the project centers on pipelines and functions.*

## 11. Technology stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| ML | scikit-learn, XGBoost, imbalanced-learn |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Config | PyYAML |
| Serialization | joblib |

## 12. API documentation

Authoritative request/response shapes are defined in **`app.py`** and the auto-generated **OpenAPI** docs at `/docs` when the server runs.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + model metadata |
| POST | `/predict` | Failure probability + binary decision |
| POST | `/feedback` | Record actual outcome for a `request_id` |
| GET | `/metrics` | Totals and predicted failure rate from log |
| GET | `/drift` | PSI summary vs training reference |

## 13. Future extensions

- LSTM / transformers on longer sequences  
- Multi-class failure modes  
- Remaining useful life (RUL)  
- Edge deployment  
- Web dashboard for fleet health  
- A/B testing for retrain validation  
