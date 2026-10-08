FROM python:3.13-slim@sha256:bf44cdfcb76cd3b41e879bc058fc37ec5872002ccfde7fcb765e218cde0cd79c
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements.txt
COPY metrics_aggregator.py .
USER 65532:65532
ENTRYPOINT ["python3", "/app/metrics_aggregator.py"]
HEALTHCHECK --interval=10s --timeout=5s --retries=3 CMD ["python3", "/app/metrics_aggregator.py", "--healthcheck"]
