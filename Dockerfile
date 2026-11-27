# Production image for the RouteWise web service (CPU only; no PyTorch needed at runtime).
# ROUTER_MODE=full is the default here; render.yaml sets ROUTER_MODE=light for Render's 512 MB free plan.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TEXT_MODEL_THREADS=1 PORT=8000 \
    ROUTER_MODE=full
WORKDIR /srv

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY src ./src
COPY app ./app
# router.joblib holds the stacker and the TF-IDF / Ridge components; the fine-tuned transformer
# (int8 ONNX) and its tokenizer are used in ROUTER_MODE=full only.
# reports/summary.json and reports/figures feed the results page.
COPY models/router.joblib models/text_router.onnx models/text_tokenizer.json ./models/
COPY reports/summary.json ./reports/summary.json
COPY reports/figures ./reports/figures

# folders the library expects to exist; the service itself never writes to them
RUN mkdir -p data/processed && useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
