# CPU inference image: FastAPI + built frontend on ONE origin. Weights are NOT baked in:
# mount a trained checkpoint read-only at /models/best.pt (see docker-compose.yml).
# GPU training is done on the host and is independent of Docker.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-api.txt ./
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt
COPY ml ./ml
COPY backend ./backend
COPY configs ./configs
COPY --from=web /web/dist ./frontend/dist

ENV MODEL_CHECKPOINT_PATH=/models/best.pt \
    RESULTS_DIR=/results \
    FRONTEND_DIST=/app/frontend/dist \
    INFERENCE_DEVICE=cpu \
    ALLOW_NON_TRAINED_CHECKPOINTS=false

RUN useradd --system --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
