# Testing and quality assurance

## Scope

| Component | What is exercised | How |
|-----------|-------------------|-----|
| Pipeline | `main.py` end-to-end | Run with `data/raw/ai4i2020.csv`; use `--skip-optimization` for a shorter run |
| API | All routes | HTTP calls against a running Uvicorn process; contract checks via `/docs` |
| Monitoring | `monitoring/monitor.py` | Execute after logs exist; inspect JSON report |
| Drift | `GET /drift` | Needs a prediction log with at least 100 rows and overlapping feature columns |

**Preconditions:** Python environment and dependencies installed; dataset present; at least one successful training run so `deployment.model_path` and `deployment.threshold_path` resolve.

**Completion:** `/health` succeeds; `/predict` returns a valid payload; `/metrics` and `/drift` respond consistently with empty versus populated logs.

## Test cases

| ID | Scenario | Steps | Expected |
|----|----------|-------|----------|
| TC-01 | Health | `GET /health` | `status`, `model`, `threshold`, `expected_features` present |
| TC-02 | Valid prediction | `POST /predict` with `Type` in {L,M,H} and numeric fields | Probability in [0,1]; binary flag; `request_id` present |
| TC-03 | Invalid category | `POST /predict` with invalid `Type` | HTTP 422 |
| TC-04 | Feedback | `POST /feedback` with known `request_id` | Acknowledgement with recorded status |
| TC-05 | Metrics, empty log | No file or empty log at configured path | Zero totals or documented empty behaviour |
| TC-06 | Metrics, with data | After several predictions | Non-zero count; rate in [0,1] |
| TC-07 | Drift, no log | Missing predictions file | `insufficient_data` (or equivalent) |
| TC-08 | Drift, short log | Fewer than 100 rows | `insufficient_data` |
| TC-09 | Drift, adequate sample | ≥100 rows, shared features with training | `avg_psi`, `drift_detected`, feature breakdown |
| TC-10 | Pipeline regression | `python main.py --skip-optimization` | Completes without error; artifacts updated |

## Automation

GitHub Actions (`.github/workflows/ci.yml`) installs dependencies and runs `python -m compileall` on the application and library code. Add `pytest` or integration tests under `tests/` when you need stricter regression coverage.

## Defect log

Record defects in your issue tracker or in the table below during development.

| ID | Summary | Severity | State |
|----|---------|----------|-------|
| — | — | — | — |

## Example `curl` (manual)

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"asset_id":"line-1","Type":"L","Air_Temp":300,"Process_Temp":310,"Rotational_Speed":1500,"Torque":40,"Tool_Wear":100}'
```

On Windows PowerShell you can use the same JSON in `Invoke-RestMethod` or a single-line `curl` with proper quoting.

Use the returned `request_id` when calling `POST /feedback`.
