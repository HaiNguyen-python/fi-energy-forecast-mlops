Data: Synthetic data. Holdout: last 30 days.

| Model | Holdout MAE (MWh) | Holdout MAPE |
| --- | --- | --- |
| Seasonal naive (same hour last week) | 529.8 | 5.18% |
| LightGBM | 112.8 | 1.10% |

| 80% prediction interval | Coverage of actual hours (target 80%) | Mean width (MWh) |
| --- | --- | --- |
| Raw quantile models (P10/P90) | 71% | 375 |
| + conformal calibration | 83% | 458 |
