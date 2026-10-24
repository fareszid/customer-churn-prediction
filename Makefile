.PHONY: install data train test lint format notebook serve docker-build docker-run

install:  ## Install the package with dev dependencies
	pip install -e ".[dev]"

data:  ## Download the raw dataset
	python -m churn.data

train:  ## Tune, compare and train models; writes models/ and reports/
	python -m churn.train --trials 60

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

notebook:  ## Re-run the EDA notebook
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratory_analysis.ipynb

serve:  ## Run the API locally on http://localhost:8000/docs
	uvicorn churn.api:app --reload

docker-build:
	docker build -t churn-api .

docker-run:
	docker run --rm -p 8000:8000 churn-api
