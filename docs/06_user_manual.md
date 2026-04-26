# User manual — predictive maintenance API

Audience: operators integrating the service and engineers running it locally. Implementation detail is in `README.md` and `docs/03_system_design.md`.

## Purpose

The service turns a single sensor snapshot (plus optional failure flags) into:

- `failure_probability` — score between 0 and 1  
- `predicted_failure` — 1 if the score meets or exceeds the deployed threshold  
- `request_id` — identifier for logging and optional feedback  

## Setup

1. Install Python 3.10 or newer.  
2. Clone the repository and run `pip install -r requirements.txt`.  
3. Place the AI4I CSV at `data/raw/ai4i2020.csv`.  
4. Train once: `python main.py`, or `python main.py --skip-optimization` for a faster pass.  
5. Confirm `config/config.yaml` points to existing files for `deployment.model_path` and `deployment.threshold_path`.

## Starting the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

- Health: `http://127.0.0.1:8000/health`  
- Swagger UI: `http://127.0.0.1:8000/docs`

## `POST /predict`

Content-Type: `application/json`.

| Field | Type | Note |
|-------|------|------|
| `asset_id` | string | Per-machine id for rolling features; default if omitted |
| `Type` | string | `L`, `M`, or `H` only |
| `Air_Temp`, `Process_Temp` | number | Kelvin |
| `Rotational_Speed` | number | rpm |
| `Torque` | number | Nm |
| `Tool_Wear` | number | minutes |
| `TWF` … `RNF` | 0 or 1 | Optional; default 0 |

Example response:

```json
{
  "request_id": "b2c9f1a0-…",
  "model": "RandomForestClassifier",
  "threshold": 0.48,
  "failure_probability": 0.61,
  "predicted_failure": 1
}
```

`predicted_failure: 1` means the current policy would raise an alert. `failure_probability` supports ranking assets or adjusting sensitivity without retraining.

Each successful call appends one line to the path in `deployment.log_path` (default under `outputs/monitoring/`).

## `POST /feedback`

After the true outcome is known:

```json
{
  "request_id": "<value from predict>",
  "actual_failure": 0
}
```

`actual_failure` must be 0 or 1. Rows go to `deployment.feedback_log_path`.

## `GET /metrics`

Returns totals from the prediction log, including count and mean predicted failure rate.

## `GET /drift`

Compares logged feature distributions to the training reference using PSI. With no log or fewer than 100 predictions, the service returns an insufficient-data message. Otherwise expect average PSI, a drift flag, and the strongest contributors.

If `drift_detected` is true, review data quality and consider `monitoring/monitor.py` (and retrain if your process allows).

## Batch monitoring

```bash
python monitoring/monitor.py
python monitoring/monitor.py --retrain --skip-optimization
```

Thresholds such as `psi_threshold` and `recall_threshold` are in `config/config.yaml` under `monitoring`.

## Troubleshooting

| Issue | Likely cause | Action |
|-------|----------------|--------|
| Import errors | Missing packages | `pip install -r requirements.txt` |
| Model load failure | Wrong path or no training | Run `python main.py`; verify YAML |
| 422 on predict | Invalid `Type` or malformed JSON | Fix body against `/docs` |
| Drift never ready | Too few logged predictions | Generate more traffic or lower the threshold only if appropriate |
| Odd rolling behaviour | Mixed streams on one `asset_id` | One stable `asset_id` per physical asset |

## Security

The sample binds to all interfaces for lab use. Production deployments should use TLS, network policy, authentication, and managed secrets; those are not part of this codebase.
