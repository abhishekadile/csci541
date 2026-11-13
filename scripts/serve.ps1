# Production-style, single origin: build the frontend, then let FastAPI serve UI + API on :8000.
# Run from the wafer-inspection folder:  .\scripts\serve.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
Push-Location frontend
npm ci --no-audit --no-fund
npm run build
Pop-Location
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
