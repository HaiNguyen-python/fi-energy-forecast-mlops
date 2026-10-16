# Learning guide (for me, Hai)

This file is my study guide for this repo. Goal: be able to explain every part in an interview,
change it confidently, and answer "why?" three levels deep. Work through it before pinning the repo.

Start with [docs/how-it-works.md](docs/how-it-works.md) (the code walkthrough) and
[docs/decisions.md](docs/decisions.md) (the reasons). This file is the practice plan.

## 1. Run it and watch each step (1 hour)

```bash
make install
energy-forecast ingest --source synthetic   # writes data/raw/hourly.parquet
energy-forecast train                        # prints metrics and the promotion decision
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://localhost:5000
energy-forecast monitor                      # writes reports/monitor_*.json
energy-forecast report                       # writes docs/results.md and docs/img/*.png
energy-forecast serve                        # open http://localhost:8000/docs
```

Run `energy-forecast train` a second time. The new model is *not* promoted. Find the reason in the
output and in the MLflow run tag `promotion_decision`.

## 2. Read the code in this order

| File | What to understand | Question to answer yourself |
| --- | --- | --- |
| `data/synthetic.py` | How demand depends on temperature, hour, weekday, holidays | Why is heating load `max(0, 17 - temp)`? |
| `data/store.py` | Upsert by timestamp | Why does idempotency matter for a daily DAG? |
| `features.py` | Lags ≥ 24h, rolling means of the *shifted* series, local time | What would leak if `roll_mean_24` used `y` instead of `y.shift(24)`? |
| `train.py` | Holdout split, `TimeSeriesSplit(gap=24)`, baseline, MLflow logging | Why not `train_test_split(shuffle=True)`? |
| `intervals.py` | Quantile (pinball) loss, conformal offset, sorting crossed quantiles | Why did raw P10/P90 cover only 71%? |
| `registry.py` | `should_promote` and the `@champion` alias | How do you roll back a bad model? |
| `serve.py` | Dependency injection (`Depends`) for model and history | Why does that make the API easy to test? |
| `monitor.py` | PSI and the seasonal reference window | Why does a week in December "drift" vs. October? |
| `dags/energy_forecast_dag.py` | Thin DAG calling the CLI | Why keep logic out of the DAG file? |

## 3. Key concepts in my own words (fill these in)

- **Data leakage:** ...
- **Seasonal naive baseline:** ...
- **MAPE vs MAE:** MAPE is relative (good for comparing periods), MAE is in MWh (good for cost). ...
- **PSI:** ...
- **Model registry and aliases:** ...
- **Quantile regression (pinball loss):** ...
- **Conformal calibration:** ...  (widen the interval by how much it missed on a calibration window)

## 4. Interview questions this repo answers

1. *How do you put a model into production?* Walk through the diagram: ingest, features, train,
   registry, promotion rule, API, monitoring, daily DAG.
2. *How do you know a model is getting worse?* Daily MAPE against actuals with a threshold (action),
   PSI drift on inputs (early warning). Explain why drift alone doesn't fail the run.
3. *How do you avoid leakage in time series?* 24-hour shift rule + the tampering unit test.
4. *Why LightGBM?* Fast on tabular data, handles non-linear temperature effects, works well with
   lag features; always compared with a baseline.
5. *How do you give a range, not just a number?* Quantile models + conformal calibration;
   show the 71% → 83% coverage table and explain the width trade-off.
6. *How would you scale this?* Partitioned storage (S3), a feature store if several models share
   features, batch inference job writing forecasts to a table, MLflow on managed infrastructure.

## 5. Make it mine (do at least three)

- [ ] Get a Fingrid API key, ingest real data, update the Results table with real numbers.
- [ ] Add a temperature *forecast* from FMI for tomorrow and compare errors.
- [ ] Calibrate the interval per hour of day and compare coverage at peak hours.
- [ ] Re-run `energy-forecast report` on real data and replace the README numbers and charts.
- [ ] Run `make up` with Docker and record a 60-second demo GIF.
- [ ] Add a feature (e.g. Helsinki + Oulu average temperature) and show its effect in MLflow.

## 6. Quality checks I should be able to explain

- `make lint` (ruff), `mypy src` (types), `pytest` (29 tests, coverage report), `pre-commit install`.
- Why the CLI test uses 400 days of data: trees can't extrapolate to temperatures they never saw.

## 7. Honesty checklist before pushing

- [ ] I can explain every file without looking at it.
- [ ] Results in the README are from runs I did myself.
- [ ] Anything I haven't built is under "What I'd do next".

Note: `docker compose up` was not tested in the environment this repo was first built in (no access
to Docker Hub there). Run `make up` once yourself and fix anything that breaks before you demo it.
