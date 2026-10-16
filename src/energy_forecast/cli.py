"""Command line entry point: ingest, train, monitor, serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime, timedelta

from .config import get_settings
from .data import store, synthetic

log = logging.getLogger("energy_forecast")


def cmd_ingest(args) -> None:
    settings = get_settings()
    if args.source == "synthetic":
        df = synthetic.generate(start=args.start, days=args.days)
    else:
        from .data import fingrid, fmi

        if not settings.fingrid_api_key:
            sys.exit("Set FINGRID_API_KEY (free at https://data.fingrid.fi) for --source real")
        end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        cons = fingrid.fetch_consumption(
            start, end, settings.fingrid_api_key, settings.fingrid_dataset_id
        )
        temp = fmi.fetch_temperature(start, end, settings.fmi_place)
        df = store.merge_sources(cons, temp)
    result = store.upsert(settings.raw_path, df)
    log.info(
        "stored %d rows (%s to %s) in %s",
        len(result),
        result["timestamp"].min(),
        result["timestamp"].max(),
        settings.raw_path,
    )


def cmd_train(args) -> None:
    from . import train

    settings = get_settings()
    result = train.run(store.load(settings.raw_path), settings)
    print(json.dumps(result.__dict__, indent=2, default=str))


def cmd_monitor(args) -> None:
    import mlflow

    from . import monitor
    from .features import training_frame
    from .registry import load_champion

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = load_champion(settings.model_name, settings.champion_alias)
    if model is None:
        sys.exit("No champion model yet. Run `energy-forecast train` first.")
    frame = training_frame(store.load(settings.raw_path))
    report = monitor.build_report(frame, model, settings.mape_alert_threshold)
    path = monitor.write_report(report, settings.reports_dir)
    print(json.dumps(report, indent=2))
    log.info("report written to %s", path)
    # Drift is a signal to investigate; only a real forecast-error breach fails the run.
    if report["drift"]["overall_status"] == "alert":
        log.warning("input drift detected: %s", report["drift"]["features"])
    if args.fail_on_alert and report["last_day_error"]["alert"]:
        sys.exit(2)


def cmd_report(args) -> None:
    from pathlib import Path

    from . import report

    settings = get_settings()
    bt = report.write(store.load(settings.raw_path), Path(args.out), args.data_label)
    print(report.results_markdown(bt, args.data_label))


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("energy_forecast.serve:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="energy-forecast")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="load data into the local store")
    p.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--start", default="2024-01-01", help="start date for synthetic data")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("train", help="train, evaluate and maybe promote a model")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("monitor", help="write a drift and error report")
    p.add_argument("--fail-on-alert", action="store_true")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("report", help="backtest charts and results table for the README")
    p.add_argument("--out", default="docs")
    p.add_argument("--data-label", default="Synthetic data")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("serve", help="run the forecast API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
