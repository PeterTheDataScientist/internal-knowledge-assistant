.PHONY: help install run test eval eval-judge smoke docker lint clean

help:
	@echo "install     install dependencies"
	@echo "run         start the API on http://localhost:8000"
	@echo "lint        run ruff over the source"
	@echo "test        run the unit and API tests"
	@echo "eval        run the evaluation suite"
	@echo "eval-judge  run the evaluation suite with the model judge"
	@echo "smoke       curl the running service with a few representative questions"
	@echo "docker      build and start the container"

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

lint:
	ruff check app eval tests scripts streamlit_app.py

test:
	pytest

eval:
	python eval/run_eval.py

eval-judge:
	python eval/run_eval.py --judge --delay 5

smoke:
	bash scripts/smoke_test.sh

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .index .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
