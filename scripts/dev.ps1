# Local development: API with auto-reload on :8000 and the Vite dev server on :5173 (proxies /api).
# Run from the wafer-inspection folder:  .\scripts\dev.ps1
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
Write-Host "API:      http://127.0.0.1:8000/api/model"
Write-Host "Frontend: http://localhost:5173"
