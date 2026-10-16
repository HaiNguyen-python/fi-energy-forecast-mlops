# How it works

This page follows one forecast from raw data to an API response, and explains the important code
on the way. File references point to `src/energy_forecast/`.

## 1. Ingest: raw data into one hourly table

`cli.py → cmd_ingest` loads data from one of two sources:

- **Real:** `data/fingrid.py` pages through the Fingrid Open Data API (and backs off on HTTP 429
  rate limits); `data/fmi.py` fetches temperature from the FMI WFS service in weekly chunks,
  because that service limits how much one request may return.
- **Synthetic:** `data/synthetic.py` builds two years of Finnish-like demand. Demand rises when it
  is cold (`140 * max(0, 17 - temperature)`), follows daily and weekly cycles, and drops on
  Finnish public holidays.

`data/store.py → upsert` writes one parquet table with `timestamp`, `consumption_mwh` and
`temperature_c`. Rows are keyed by timestamp, so running the same day twice replaces rows instead
of duplicating them. That is what makes the daily Airflow job safe to retry.

## 2. Features: only what is known at forecast time

`features.py → build_features` turns the table into model inputs:

| Feature | Meaning |
| --- | --- |
| `lag_24`, `lag_48`, `lag_168` | demand 1 day, 2 days and 1 week earlier |
| `roll_mean_24`, `roll_std_24`, `roll_mean_168` | level and volatility, computed on the series *shifted by 24 h* |
| `hour`, `dayofweek`, `month`, `is_weekend` | calendar, in Helsinki local time |
| `is_holiday` | Finnish public holidays (`holidays` package) |
| `temperature_c`, `hdd` | temperature and heating degrees `max(0, 17 - temp)` |

The rule that matters: a forecast for tomorrow is made with data up to the end of today, so every
demand-based feature is shifted by at least 24 hours. `tests/test_features.py →
test_no_leakage_when_future_values_change` multiplies the last day of demand by 10 and checks that
the features for that day do not move. If someone later adds a leaky feature, this test fails.

`future_frame` builds the 24 rows for a day that has not happened yet: it appends empty rows for
tomorrow to the recent history and runs the same `build_features`, so training and serving can
never compute features differently.

## 3. Train: compare, validate, register

`train.py → run`:

1. **Hold out** the last 30 days (`split_holdout`). Nothing in that period touches training,
   model selection or calibration.
2. **Cross-validate** on the rest with `TimeSeriesSplit(n_splits=5, gap=24)`: every fold trains
   on the past and validates on the future, never the other way round.
3. **Fit LightGBM** on all training data and score it on the holdout.
4. **Score the baseline** ("same hour last week" = `lag_168`) on the same holdout.
5. **Fit the interval model** (`intervals.py`, next section).
6. **Log everything** to MLflow: parameters, data range, CV metrics, holdout metrics, the models.
7. **Decide promotion** with `registry.py → should_promote`.

## 4. Prediction intervals: quantiles plus a calibration step

`intervals.py` adds an 80% interval around each forecast.

- Two extra LightGBM models use the quantile (pinball) loss with `alpha=0.1` and `alpha=0.9`.
- On their own they were over-confident: on the backtest only 71% of actual hours fell inside the
  "80%" interval.
- **Conformal calibration** fixes this. The quantile models are trained without the last 30 days
  of the training period. On those 30 days we measure how far actual values land outside the raw
  interval (`conformal_offset`) and widen the interval by that amount. Coverage on the unseen
  holdout becomes 83%.
- `predict_interval` also sorts P10, point and P90 per row, because separately trained quantile
  models can "cross".

## 5. Promotion: a rule in code

```python
should_promote(candidate_mape, baseline_mape, champion_mape, min_improvement=0.01)
```

A new model becomes the champion only if it beats the baseline **and** beats the current champion's
MAPE by at least 1% (relative), on the same holdout. The decision and its reason are stored as an
MLflow tag. Promotion moves the MLflow alias `@champion`; rollback means pointing the alias at an
older version. `tests/test_end_to_end.py` trains twice and checks that the identical second model
is *not* promoted.

## 6. Serve: FastAPI

`serve.py`:

- `get_model`, `get_interval_model` and `get_history` are FastAPI dependencies. The real app loads
  the champion from MLflow; tests swap in a trained model with `app.dependency_overrides`, so the
  API is tested without a running MLflow server.
- `POST /forecast {"date": "2025-12-31"}` returns 24 hours with `forecast_mwh`, `p10_mwh`, `p90_mwh`.
  Optional `temperatures` lets a caller pass a weather forecast; otherwise last week's
  temperatures are used as a simple fallback.
- Bad input (wrong date, wrong number of temperatures, stale history) returns HTTP 422 with a
  readable message instead of a crash.

## 7. Monitor: drift and error

`monitor.py → build_report` writes a JSON report every day:

- **Input drift** with the Population Stability Index (PSI) on temperature and demand features.
  Demand is seasonal, so last week is compared with the *same weeks one year earlier*, not with
  the previous months (which would always look like drift in autumn).
- **Forecast error** on the latest day of actuals, against an 8% MAPE threshold.
- Drift is logged as a warning (a reason to look); only an error breach fails the Airflow task.

## 8. Orchestrate: Airflow

`dags/energy_forecast_dag.py` runs `ingest → train → monitor` daily at 06:00 UTC with two retries.
The DAG only calls the CLI. All logic lives in the package, where it is unit tested.

## 9. Report: charts in this README

`report.py` (`energy-forecast report`) re-runs the backtest without MLflow and writes
`docs/results.md` and the two charts. Chart titles are computed from the numbers, so they stay
true when the report is re-run on real Fingrid data.
