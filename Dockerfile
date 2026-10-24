# Serves the trained model. Build after training: `make train && make docker-build`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHURN_PROJECT_ROOT=/app

# libgomp1 is the OpenMP runtime XGBoost needs on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY models ./models
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
