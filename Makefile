.PHONY: install test lint typecheck format up down demo report

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

report:
	energy-forecast report

format:
	ruff format .
	ruff check --fix .

# Full local loop without Docker: synthetic data -> train -> monitor -> API on :8000
demo:
	energy-forecast ingest --source synthetic
	energy-forecast train
	energy-forecast monitor
	energy-forecast serve

up:
	docker compose up --build

down:
	docker compose down -v
