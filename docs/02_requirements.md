# Requirements

## 1. Stakeholder analysis

| Stakeholder | Interest | Expectations |
|-------------|----------|--------------|
| **Maintenance / operations** | Reduce unplanned downtime | Timely failure alerts; interpretable signals (e.g. tool wear, torque variability) |
| **Data / ML engineers** | Reliable pipeline & deployment | Config-driven training; versioned artifacts; parity between batch and online features |
| **IT / platform** | Runnable service | Documented install, ports, dependencies; health checks |
| **Course assessors** | Traceable design & testing | Requirements, design artifacts, test evidence, GitHub documentation |

## 2. User stories

1. **As a** maintenance analyst, **I want** to send current sensor readings to an API **so that** I receive a failure probability and alert flag without running a notebook.
2. **As an** ML engineer, **I want** training to be one command from config **so that** results are reproducible and comparable across runs.
3. **As an** operator, **I want** to record actual outcomes after a prediction **so that** online precision/recall can be tracked.
4. **As a** reliability engineer, **I want** drift detection against training distributions **so that** model refresh can be triggered before performance collapses.

## 3. Use cases (summary)

| ID | Actor | Goal |
|----|-------|------|
| UC-1 | Client system | Submit sensor event → receive `failure_probability`, `predicted_failure`, `request_id` |
| UC-2 | Operator | Submit feedback with `request_id` and `actual_failure` |
| UC-3 | Ops / engineer | Check service and model metadata |
| UC-4 | Ops / engineer | Inspect aggregate prediction metrics |
| UC-5 | Ops / engineer | Run drift check (PSI) over recent logs |

Detailed flows appear as sequence diagrams in [03_system_design.md](03_system_design.md).

## 4. Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Load raw AI4I CSV, drop non-predictive IDs, rename columns per config | Must |
| FR-2 | Engineer physics and rolling-window features consistent with training | Must |
| FR-3 | Train and evaluate at least Random Forest and XGBoost; persist best artifact | Must |
| FR-4 | Expose `GET /health` with model name, threshold, feature count | Must |
| FR-5 | Expose `POST /predict` accepting validated sensor payload (see OpenAPI / `app.py`) | Must |
| FR-6 | Append each prediction to CSV log with timestamp and `request_id` | Must |
| FR-7 | Expose `POST /feedback` to log actual labels | Should |
| FR-8 | Expose `GET /metrics` for volume and predicted failure rate from logs | Should |
| FR-9 | Expose `GET /drift` for PSI-based comparison vs training reference | Should |
| FR-10 | Monitoring script reads logs and config thresholds; optional `--retrain` | Should |

## 5. Non-functional requirements

| ID | Category | Requirement |
|----|----------|---------------|
| NFR-1 | **Config** | Hyperparameters and paths live in `config/config.yaml`, not hardcoded in business logic |
| NFR-2 | **Integrity** | Train/test split does not shuffle rows; SMOTE applies only to training split |
| NFR-3 | **Robustness** | API validates payload types/ranges where defined (Pydantic); errors return HTTP 500 with logged exception |
| NFR-4 | **Observability** | Predictions and feedback persisted to CSV under `outputs/monitoring/` |
| NFR-5 | **Security (course / demo)** | No authentication in default demo; production would add TLS, authn/z, and secret management |
| NFR-6 | **Portability** | Python 3.10+ recommended; dependencies pinned in `requirements.txt` |

## 6. UI / UX

There is **no end-user graphical UI** in this repository; interaction is via **HTTP API** (and optional `curl`/Postman). For course rubrics that require UI artifacts, use:

- **API as the “interface”:** example requests in [06_user_manual.md](06_user_manual.md).
- **Optional:** a single wireframe of a future “fleet dashboard” is described in [03_system_design.md](03_system_design.md) as future work.
