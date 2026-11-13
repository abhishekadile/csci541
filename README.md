# Wafer Defect Inspection (CSCI 441/541)

Six-class classification of full-resolution Carinthia-S SEM images, with a **visual inspection PASS/FAIL**
decision derived from the predicted class, a FastAPI inference service, and a React inspection workstation.

* **Task.** One SEM inspection site (480 x 480, grayscale) goes in. A source class 1-6 comes out.
* **Decision rule** (one function, `ml/decision.py`, used by evaluation, CLI and API):
  predicted class **6 -> PASS** ("no visible defect"); predicted class **1-5 -> FAIL** ("visible defect").
* **Scope.** This is a visual inspection of the imaged site. The data does **not** establish electrical die yield
  or the disposition of an entire wafer, and the software makes no such claim.
* **Models.** Both are ResNet-18 trained **from scratch** (no pretrained weights are ever downloaded), one input
  channel, six logits, native 480 x 480 input.
  * **Model A (baseline):** class labels only.
  * **Model B (mask-supervised):** same encoder and head plus a light segmentation decoder, trained with the expert
    masks (`loss = class-weighted cross-entropy + aux_weight * (BCE + soft Dice)`, `aux_weight = 1.0`).
    At inference both take only the image; B also returns a predicted defect mask.

```
wafer-inspection/
  ml/         audit, split, data, models, losses, train, evaluate, compare, predict, decision, smoke_test, benchmark, run_all
  configs/    common.yaml (shared by both models), model_a.yaml, model_b.yaml, smoke.yaml
  backend/    FastAPI service (backend/app/main.py)
  frontend/   React + TypeScript + Vite (Inspection and Experiments views)
  tests/      pytest suite (labels, decision, metrics, data, checkpoints, API)
  scripts/    setup.ps1, dev.ps1, serve.ps1, trace_sample.py
  docs/       experiment-design.md, report-outline.md, deployment.md
  artifacts/  generated manifests, checkpoints, metrics, plots (git-ignored)
```

The dataset stays **outside** this folder and is referenced through `DATA_ROOT`. Images, weights, credentials and
generated outputs are git-ignored.

## 0. Dataset (important)

The code needs **Carinthia-S** (images **and** expert masks): <https://zenodo.org/records/16895427> (CC BY 4.0,
`data.zip`, 139.2 MB, md5 `3a41242574635cf1302fab15accb7730`). The original *Carinthia* archive (`carinthia.csv`, images
only, no masks) is **not** enough and the code refuses it.

On this machine Carinthia-S is already extracted to
`D:\SCSU Classes material\Masters\CSCI 541\Project\data-carinthia-s\extracted\data` (contains `images/`, `masks/`,
`carinthia-s.csv`). To fetch it elsewhere:

```powershell
$dst = "D:\datasets\carinthia-s"; New-Item -ItemType Directory -Force $dst | Out-Null
curl.exe -L --fail -o "$dst\data.zip" "https://zenodo.org/records/16895427/files/data.zip?download=1"
(Get-FileHash "$dst\data.zip" -Algorithm MD5).Hash   # must be 3A41242574635CF1302FAB15ACCB7730
Expand-Archive "$dst\data.zip" -DestinationPath "$dst\extracted"
```

## 1. Setup (copy-paste, in order)

```powershell
cd "D:\SCSU Classes material\Masters\CSCI 541\Project\wafer-inspection"

# If PowerShell blocks the script: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup.ps1          # uv venv + CUDA torch 2.7.1 (cu126) + Python deps + npm install; add -Cpu for CPU-only torch, -Recreate to rebuild .venv
.\.venv\Scripts\Activate.ps1

# Check the RTX 2080 is visible to PyTorch (expect: True NVIDIA GeForce RTX 2080)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Configure `DATA_ROOT`

```powershell
# Edit .env (created by setup.ps1) or set it for this shell only:
$env:DATA_ROOT = "D:\SCSU Classes material\Masters\CSCI 541\Project\data-carinthia-s\extracted\data"
```

## 3. Audit the data and build the split

```powershell
python -m ml.audit        # paths, dimensions, class counts, masks, label/mask disagreements, duplicates (about 2 min the first time)
python -m ml.split        # 70/15/15 manifest, train-only normalization and defect-size cutoffs; both models share it
```

## 4. Smoke test and hardware benchmark (neither is training)

```powershell
python -m ml.smoke_test   # real image/mask pairs, forward + backward for BOTH models; output marked smoke
python -m ml.benchmark    # times train/val at 480 x 480, peak GPU memory, estimates a full run (about 1 min)
```

Smoke/benchmark outputs go to `artifacts/smoke/`. Smoke checkpoints carry `provenance: "smoke"` and the loader and API
**refuse** them as production models.

## 5. Train both models, evaluate, compare (long)

```powershell
python -m ml.run_all                      # split -> train A -> train B -> evaluate val + test -> comparison
# Interrupted? Resume exactly where it stopped:
python -m ml.run_all --resume
# GPU out of memory? Keep full resolution, lower the batch size:
python -m ml.run_all --batch-size 4       # or 2
```

Or step by step:

```powershell
python -m ml.train model_a.yaml           # baseline  -> weight\model_a\{best,latest}.pt; logs in artifacts\runs\model_a
python -m ml.train model_b.yaml           # mask-supervised -> weight\model_b\...; logs in artifacts\runs\model_b
python -m ml.train model_a.yaml --resume  # continue after an interruption (same for model_b.yaml)
python -m ml.evaluate model_a.yaml --split val
python -m ml.evaluate model_b.yaml --split val
python -m ml.evaluate model_a.yaml --split test   # once per frozen checkpoint (refuses to repeat without --force)
python -m ml.evaluate model_b.yaml --split test
python -m ml.compare                      # artifacts\results\comparison.json/.md, plots, error gallery
```

Training settings are in `configs/common.yaml`: batch size 8, up to 50 epochs, early stopping (patience 10), AdamW,
FP16 autocast on CUDA, seed 42. Model selection (both models, same rule): highest validation macro-F1, ties broken
by lower validation cross-entropy. The test split is never used for training or selection.

Measured on the RTX 2080 (`python -m ml.benchmark`, batch 8, 480 x 480, FP16): see the table in
`docs/experiment-design.md` section 7. Use it as an estimate; real epochs were slower than the short benchmark.

## 6. Run the app

```powershell
# Development: API on :8000 (auto-reload) + Vite on :5173
.\scripts\dev.ps1                          # then open http://localhost:5173

# Production-style, ONE origin: build the frontend, FastAPI serves UI + API on :8000
.\scripts\serve.ps1                        # then open http://127.0.0.1:8000
```

Choose which trained model the service loads (it loads once at startup, only from this configured path):

```powershell
$env:MODEL_CHECKPOINT_PATH = "weight/model_b/best.pt"   # or model_a for the classification-only baseline
```

The service starts even when no checkpoint exists: `/api/model` reports why it is not ready, `/api/predict` answers 503,
and the Experiments view shows the command that generates missing results. Nothing is ever faked.

API summary (`/docs` for the OpenAPI UI):

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | liveness |
| `GET /api/model` | readiness, reason when not ready, model metadata, class list, decision rule |
| `POST /api/predict` | multipart `file` (JPEG/PNG); returns `predicted_class_id`, `predicted_class_name`, `predicted_class_score`, `class_scores` (keys "1".."6"), `inspection_decision`, `decision_reason`, `model_version`, original size, `processing_time_ms`, and `predicted_mask` (Model B only) |
| `GET /api/experiments` | comparison artifacts, or `available: false` plus the generating command |

Uploads are validated (real image content, JPEG/PNG only, size limit, exact trained dimensions), are processed in memory,
and are never written to disk.

## 7. Verify (tests, type check, production build)

```powershell
python -m pytest                            # 60 tests: label mapping, decision rule, metrics, data, checkpoints, API
cd frontend; npm run typecheck; npm run build; cd ..
python scripts\trace_sample.py --class-id 6 --checkpoint weight\model_b\best.pt   # real sample -> API -> display fields
```

## 8. Docker (CPU inference only; training stays on your GPU, outside Docker)

```powershell
docker compose up --build                   # http://localhost:8000, checkpoint and results mounted read-only
```

See `docs/deployment.md`.

## Troubleshooting

**CUDA not detected** (`torch.cuda.is_available()` is `False`)
* `python -c "import torch; print(torch.__version__)"` should end in `+cu126`. A `+cpu` suffix means the CPU wheel is installed.
  Reinstall: `uv pip install --python .venv\Scripts\python.exe --reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match` (or just `.\scripts\setup.ps1 -Recreate`)
* Run `nvidia-smi`; if it fails, update the NVIDIA driver. Make sure you activated `.venv` (the system Python may have the CPU build).

**CUDA out of memory**
* Lower the batch size and keep the resolution: `python -m ml.run_all --batch-size 4` (or `2`). Close other GPU programs
  (browsers, games). Model A needs about 0.6 GB and Model B about 1.6 GB at batch 8 in the benchmark.
* Resume after a crash with `--resume`.

**Missing checkpoint / "Model unavailable"**
* The API loads only `MODEL_CHECKPOINT_PATH` (default `weight/model_b/best.pt`). Train first (section 5) or point it at an existing file.
* "Refusing to load a 'smoke' checkpoint": smoke-test outputs are intentionally not loadable. Train a real model.
* "...older binary visible-defect model": checkpoints from the old binary version are rejected. Retrain.

**`DATA_ROOT is not set` / "No Carinthia-S index found"**: see sections 0 and 2. The folder must contain `carinthia-s.csv`, `images/` and `masks/`.

**Slow or hanging data loading on Windows**: keep `num_workers: 0` in `configs/common.yaml` (default), or try `--num-workers 2`.

## Honest limitations

See `docs/experiment-design.md` section 11. In short: image-level split (no wafer IDs), class 2 and class 5 have only
1 test image each, class 6 is dominated by a few near-identical clusters, and one seed is one sample.
