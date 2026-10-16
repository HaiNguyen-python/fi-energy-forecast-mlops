"""Daily Airflow DAG: ingest -> train (+ promotion) -> monitor.

Runs at 06:00 UTC, after Fingrid has published the previous day.
Each task calls the same CLI used locally, so the DAG holds no business logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

DEFAULT_ARGS = {"retries": 2, "retry_delay": timedelta(minutes=10)}


@dag(
    dag_id="fi_energy_forecast",
    schedule="0 6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "forecasting"],
)
def fi_energy_forecast():
    @task
    def ingest():
        from energy_forecast.cli import main

        # Re-fetching the last 14 days is safe: the store upserts by timestamp.
        main(["ingest", "--source", "real", "--days", "14"])

    @task
    def train():
        from energy_forecast.cli import main

        main(["train"])

    @task
    def monitor():
        from energy_forecast.cli import main

        # Exit code 2 (forecast error above threshold) fails the task and triggers alerting.
        # Drift is logged as a warning in the report.
        main(["monitor", "--fail-on-alert"])

    ingest() >> train() >> monitor()


fi_energy_forecast()
