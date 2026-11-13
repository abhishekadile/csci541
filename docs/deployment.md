# Deployment guide

Training happens on your GPU laptop, outside Docker. Deployment is CPU inference of one trained checkpoint, with the
frontend and API on **one origin**: FastAPI serves the built frontend, so there is no CORS configuration.

## What gets deployed

| Piece | Source | Notes |
|---|---|---|
| API + UI | `backend/`, `ml/`, `frontend/dist` | one process, one port (8000) |
| Checkpoint | `weight/model_a/best.pt` or `model_b/best.pt` | trained by `python -m ml.run_all`; mounted read-only |
| Results (optional) | `artifacts/results/` | produced by `python -m ml.compare`; powers the Experiments view |

A CUDA-trained checkpoint loads on CPU (`torch.load(..., map_location="cpu", weights_only=True)`). Only a checkpoint with
`provenance == "trained"` is accepted. Smoke-test and test-fixture checkpoints, and old binary checkpoints, are refused.

## Option 1: local, single origin (no Docker)

```powershell
cd "D:\SCSU Classes material\Masters\CSCI 541\Project\wafer-inspection"
.\.venv\Scripts\Activate.ps1
$env:MODEL_CHECKPOINT_PATH = "weight/model_b/best.pt"
.\scripts\serve.ps1          # npm ci + npm run build + uvicorn on http://127.0.0.1:8000
```

## Option 2: Docker (CPU, read-only mounts)

```powershell
# After training, choose the model to serve in docker-compose.yml (default: model_b), then:
docker compose up --build
# http://localhost:8000   health: http://localhost:8000/api/health   readiness: http://localhost:8000/api/model
```

The image installs the CPU build of PyTorch, runs as a non-root user, has a read-only root filesystem, and receives the
checkpoint at `/models/best.pt` and results at `/results`, both read-only. Model weights are never baked into the image.

Without a checkpoint the container still starts; `/api/model` reports `checkpoint_missing`.

## Configuration (`.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_CHECKPOINT_PATH` | `weight/model_b/best.pt` | the only place weights are loaded from |
| `RESULTS_DIR` | `artifacts/results` | evaluation artifacts for the Experiments view |
| `FRONTEND_DIST` | `frontend/dist` | built frontend served at `/` |
| `MAX_UPLOAD_BYTES` | 5242880 | upload size limit |
| `INFERENCE_DEVICE` | `cpu` | `cuda` only if available on the server |
| `ALLOW_NON_TRAINED_CHECKPOINTS` | `false` | never enable in production |

## Reverse proxy (optional)

If you front the service with nginx or Caddy, proxy everything to `127.0.0.1:8000`, keep a body-size limit of at least
`MAX_UPLOAD_BYTES`, and terminate TLS there. Uploads are processed in memory and not stored.

## Operational notes

* Inference of one 480 x 480 image on CPU took on the order of 0.2-0.3 s in the verification runs (processing time is
  returned with every prediction).
* The service holds one model; to serve both, run two instances on different ports.
* The decision shown to users is computed on the server by `ml/decision.py`; the frontend never recomputes it.
