# Production image for the RouteWise web service (CPU only; no PyTorch needed at runtime).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PORT=8000 \
    ROUTER_MODE=full FASTEMBED_CACHE=/srv/.fastembed
WORKDIR /srv

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Download the ONNX sentence-embedding model at build time, so the first request is fast.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/srv/.fastembed')"

COPY src ./src
COPY app ./app
COPY models/router.joblib ./models/router.joblib
COPY reports/summary.json ./reports/summary.json
COPY reports/figures ./reports/figures

RUN useradd --create-home appuser && chown -R appuser /srv/.fastembed
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
