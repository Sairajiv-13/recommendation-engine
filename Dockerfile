# Builds the MovieLens artifacts at image build time (the data download
# needs network access during the build) and serves the API with uvicorn.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RECS_ARTIFACTS=/app/artifacts \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

COPY data ./data
RUN python data/get_data.py --out data \
    && python -m recs.train --data-dir data/ml-latest-small --out artifacts

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/healthz', timeout=4)"
CMD uvicorn recs.api:app --host 0.0.0.0 --port ${PORT:-8000}
