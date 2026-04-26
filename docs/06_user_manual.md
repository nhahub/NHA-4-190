# User Manual — Predictive Maintenance API

This guide is for **operators** and **integrators** who need to run the service and interpret outputs. Developers should also read `README.md` and `docs/03_system_design.md`.

## 1. What the system does

The API accepts **sensor readings** (temperature, torque, speed, tool wear, etc.) and returns:

- **`failure_probability`** — estimated chance of failure (0–1).
- **`predicted_failure`** — 1 if probability ≥ configured threshold, else 0.
- **`request_id`** — unique id for logging and optional feedback.

## 2. Prerequisites

1. Python **3.10+** installed.
2. Repository cloned; dependencies installed:

   ```bash
   pip install -r requirements.txt
   ```

3. Training data placed at `data/raw/ai4i2020.csv` (see README for dataset link).
4. Model trained at least once:

   ```bash
   python main.py
   ```

   or fast debug:

   ```bash
   python main.py --skip-optimization
   ```

5. Config paths in `config/config.yaml` point to existing **`deployment.model_path`** and **`deployment.threshold_path`**.

## 3. Starting the service

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

- **Health check:** open or request `http://127.0.0.1:8000/health`.
- **Interactive docs:** `http://127.0.0.1:8000/docs` (Swagger UI).

## 4. Calling `POST /predict`

**Endpoint:** `POST /predict`  
**Content-Type:** `application/json`

**Body fields (required unless noted):**

| Field | Type | Notes |
|-------|------|--------|
| `asset_id` | string | Optional in schema default `"default"`; use one id per machine for rolling features |
| `Type` | string | Exactly `L`, `M`, or `H` |
| `Air_Temp` | number | Kelvin |
| `Process_Temp` | number | Kelvin |
| `Rotational_Speed` | number | RPM |
| `Torque` | number | Nm |
| `Tool_Wear` | number | minutes |
| `TWF` … `RNF` | integer 0/1 | Optional failure flags; default 0 |

**Example response:**

```json
{
  "request_id": "8f2c…",
  "model": "RandomForestClassifier",
  "threshold": 0.48,
  "failure_probability": 0.61,
  "predicted_failure": 1
}
```

**Interpretation:**

- **`predicted_failure: 1`** means “alert / plan intervention” under current threshold policy.
- **`failure_probability`** supports ranking multiple assets or tuning alert aggressiveness.

Each successful call **appends one row** to `outputs/monitoring/predictions_log.csv` (path from config).

## 5. Submitting feedback — `POST /feedback`

After you know the true outcome for a prediction, send:

```json
{
  "request_id": "<uuid from predict>",
  "actual_failure": 1
}
```

`actual_failure` must be `0` or `1`. This supports **online evaluation** used by the monitoring job.

## 6. Metrics — `GET /metrics`

Returns aggregate counts from the prediction log:

- **`total_predictions`**
- **`predicted_failure_rate`** — mean of binary predicted flags in the log

## 7. Drift — `GET /drift`

Compares recent prediction feature distributions to the **training reference** via **PSI**.

- If there is **no log** or **fewer than 100** predictions, the response explains insufficient data.
- Otherwise you receive **`avg_psi`**, **`drift_detected`**, and top features.

**Alerting:** When `drift_detected` is true, investigate data quality and consider retraining (see monitoring script).

## 8. Monitoring job (batch)

```bash
python monitoring/monitor.py
```

Optional:

```bash
python monitoring/monitor.py --retrain --skip-optimization
```

Read `monitoring/monitor.py` and `config/config.yaml` → `monitoring` for thresholds (`psi_threshold`, `recall_threshold`, etc.).

## 9. Troubleshooting

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| Import / module errors | Dependencies not installed | `pip install -r requirements.txt` |
| Model not found | Paths in YAML wrong or training not run | Run `python main.py`; check `deployment.model_path` |
| 422 on predict | Invalid `Type` or bad JSON | Use L/M/H only; validate body |
| Drift always insufficient | Not enough logged predictions | Generate ≥100 predictions first |
| Rolling features look odd | Mixed `asset_id` on one stream | Use consistent `asset_id` per machine |

## 10. Security note (demo vs production)

The demo server binds to `0.0.0.0` for accessibility. In production, use **TLS**, **authentication**, network restrictions, and **secrets management** — not included in this course baseline.
