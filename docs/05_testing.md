# Testing & Quality Assurance

## 1. Test plan

| Area | Scope | Approach |
|------|-------|----------|
| Training pipeline | `main.py` end-to-end | Run with real `ai4i2020.csv`; optional `--skip-optimization` for speed |
| API | FastAPI routes | Manual / scripted HTTP calls; OpenAPI `/docs` for contract checks |
| Monitoring | `monitoring/monitor.py` | Run after generating log data; verify report JSON |
| Drift | `GET /drift` | Requires ≥100 rows in prediction log |

**Entry criteria:** Python env installed; dataset in `data/raw/`; `python main.py` completed once so `models/` contains artifacts referenced by `config/config.yaml`.

**Exit criteria:** Health OK; sample prediction returns valid JSON; drift/metrics behave as documented for empty vs populated logs.

## 2. Test cases

| ID | Scenario | Steps | Expected result |
|----|----------|-------|-----------------|
| TC-01 | Health | `GET /health` | `status: ok`, `model`, `threshold`, `expected_features` present |
| TC-02 | Predict — valid payload | `POST /predict` with valid `Type` L/M/H and numeric fields | `failure_probability` in [0,1], `predicted_failure` 0 or 1, `request_id` returned |
| TC-03 | Predict — invalid Type | `POST /predict` with `Type: "X"` | HTTP 422 validation error |
| TC-04 | Feedback | `POST /feedback` with prior `request_id` | `status: recorded` |
| TC-05 | Metrics — empty log | Delete or move log file; `GET /metrics` | `total_predictions: 0` (or zero rows) |
| TC-06 | Metrics — with data | After several predictions, `GET /metrics` | `total_predictions` > 0, `predicted_failure_rate` in [0,1] |
| TC-07 | Drift — no log | No predictions file | `insufficient_data` |
| TC-08 | Drift — small log | <100 predictions | `insufficient_data` |
| TC-09 | Drift — sufficient | ≥100 predictions, overlapping features | `status: ok`, `avg_psi`, `drift_detected` |
| TC-10 | Pipeline | `python main.py --skip-optimization` | Completes without error; artifacts updated |

## 3. Automated testing

A lightweight **CI** workflow (`.github/workflows/ci.yml`) runs static checks (`compileall`) on push/PR. **Unit tests** are not required for course credit unless specified; you can add `pytest` under `tests/` later.

## 4. Bug tracking

| ID | Description | Severity | Status |
|----|-------------|----------|--------|
| — | *No formal bugs filed for submission baseline* | — | — |

Replace this table with real issues if you use GitHub Issues during the project.

## 5. Sample API calls (manual regression)

```bash
# Health
curl -s http://127.0.0.1:8000/health

# Predict (example payload — adjust numbers as needed)
curl -s -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"asset_id\":\"line-1\",\"Type\":\"L\",\"Air_Temp\":300,\"Process_Temp\":310,\"Rotational_Speed\":1500,\"Torque\":40,\"Tool_Wear\":100}"

# PowerShell-friendly (single line):
# curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"asset_id":"line-1","Type":"L","Air_Temp":300,"Process_Temp":310,"Rotational_Speed":1500,"Torque":40,"Tool_Wear":100}'
```

Use the returned `request_id` for `POST /feedback`.
