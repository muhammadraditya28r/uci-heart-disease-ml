FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY src ./src
COPY models ./models

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "heart_disease.api:app", "--host", "0.0.0.0", "--port", "8000"]
