# Requirements

## Stakeholders

| Stakeholder | Primary concern |
|-------------|-----------------|
| Maintenance and operations | Fewer unplanned stops; clear alert semantics |
| ML and data engineering | Reproducible training, versioned artifacts, identical features offline and online |
| Platform / IT | Installable service, documented ports and dependencies, health endpoint |
| Assessment | Traceable requirements, design, tests, and repository structure |

## Needs (user-oriented)

- Integrate live sensor readings without running notebooks: HTTP API with probability and binary alert.
- Re-run training from configuration: single entry point (`main.py`) and one YAML source of truth.
- Record ground truth after predictions: feedback keyed by `request_id` for later evaluation.
- Detect distribution shift: drift indicator derived from logged features versus training data.

## Use cases (summary)

| ID | Actor | Outcome |
|----|--------|---------|
| UC-1 | Calling system | Submit sensor payload; receive `failure_probability`, `predicted_failure`, `request_id` |
| UC-2 | Operator | Send `request_id` and `actual_failure` for a past prediction |
| UC-3 | Engineer | Read model and threshold metadata from `/health` |
| UC-4 | Engineer | Read volume and alert rate from `/metrics` |
| UC-5 | Engineer | Run PSI-based drift summary from `/drift` |

Interaction detail is shown in [03_system_design.md](03_system_design.md).

## Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Ingest AI4I CSV; drop identifiers per config; apply rename map | Must |
| FR-2 | Build physics and rolling-window features consistent with training | Must |
| FR-3 | Train and evaluate Random Forest and XGBoost; persist selected model | Must |
| FR-4 | Implement `GET /health` with model name, threshold, expected feature count | Must |
| FR-5 | Implement `POST /predict` with validated body per `app.py` | Must |
| FR-6 | Append each prediction to the configured CSV log | Must |
| FR-7 | Implement `POST /feedback` | Should |
| FR-8 | Implement `GET /metrics` from the prediction log | Should |
| FR-9 | Implement `GET /drift` against training reference features | Should |
| FR-10 | Batch monitoring script with optional `--retrain` | Should |

## Non-functional requirements

| ID | Topic | Requirement |
|----|--------|-------------|
| NFR-1 | Configuration | Training and deployment settings live in `config/config.yaml` |
| NFR-2 | Data integrity | No row shuffle on split; SMOTE only on training data |
| NFR-3 | API behaviour | Request validation via Pydantic; failures logged and returned as HTTP errors |
| NFR-4 | Traceability | Predictions and feedback written under `outputs/monitoring/` |
| NFR-5 | Hardening | Demo deployment has no auth; production would add TLS, access control, and secret handling |
| NFR-6 | Environment | Python 3.10+; dependencies listed in `requirements.txt` |

## User interface

There is no separate graphical application. Clients use the REST API (see [06_user_manual.md](06_user_manual.md)) or the interactive OpenAPI UI at `/docs` when the server is running. If a module requires UI mock-ups, treat the OpenAPI contract and example payloads as the interface specification, or attach a separate dashboard concept as supplementary material.
