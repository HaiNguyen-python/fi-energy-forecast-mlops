# Design decisions

Short records of the main choices: what was decided, why, and what it costs. Format inspired by
Architecture Decision Records (ADRs).

## 1. LightGBM with lag features, not a deep learning model

- **Decision:** gradient-boosted trees on lag, calendar and temperature features.
- **Why:** tabular, hourly data with strong known drivers; LightGBM trains in seconds, is easy to
  explain, and is a strong baseline in forecasting competitions.
- **Cost:** trees cannot extrapolate. Trained only on winter data, the model failed in spring
  (MAPE 6.2% on a spring holdout in a test run). Keep at least one full year of training data.
- **Alternatives:** SARIMAX (weaker with many drivers), LSTM / temporal fusion transformer
  (more data and tuning for little gain at this scale).

## 2. Always compare with a seasonal-naive baseline

- **Decision:** a model counts only if it beats "same hour last week".
- **Why:** a low error number means nothing without a reference. The baseline is free and hard to
  beat on some days (holidays, stable weather).

## 3. Time-based validation and a final holdout

- **Decision:** `TimeSeriesSplit(gap=24)` for CV plus a 30-day holdout never used for training.
- **Why:** random splits leak the future into training and give over-optimistic results.

## 4. Promotion by rule, rollback by alias

- **Decision:** `should_promote()` decides; MLflow alias `@champion` marks production.
- **Why:** reproducible, testable decisions; rollback is one alias move.
- **Cost:** a 1% minimum improvement means small real gains are skipped; that is deliberate to
  avoid churn from noise.

## 5. Conformal calibration for intervals

- **Decision:** quantile models + split conformal calibration on the last 30 training days.
- **Why:** raw quantile models covered 71% instead of 80%. Calibration brought it to 83% on unseen
  data with one extra number (the offset) to store.
- **Cost:** wider intervals (458 vs 375 MWh on average). Conformal guarantees assume the
  calibration period looks like the future; time series bend that, so coverage is re-checked on
  the holdout every training run.

## 6. Seasonal drift reference

- **Decision:** PSI against the same weeks one year earlier, not the previous months.
- **Why:** demand and temperature move with the seasons; a naive reference raises false alarms.
- **Cost:** needs a year of history (falls back to the previous 28 days otherwise).

## 7. Drift warns, error fails

- **Decision:** only forecast error above 8% MAPE fails the daily run.
- **Why:** a cold snap is real drift but the model can still be accurate. Paging people for
  harmless drift teaches them to ignore alerts.

## 8. Synthetic data in the repo

- **Decision:** ship a synthetic generator; real data needs a free Fingrid key.
- **Why:** anyone (and CI) can run everything offline in minutes.
- **Cost:** synthetic data is cleaner than reality, so README numbers are labelled as synthetic
  until replaced with real results.
