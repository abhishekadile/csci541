# One-time setup with uv. Run from the wafer-inspection folder:  .\scripts\setup.ps1   (add -Cpu for a CPU-only torch)
# Install uv first if needed:  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
param([switch]$Cpu, [switch]$Recreate)
Set-Location (Join-Path $PSScriptRoot "..")

# uv and npm print progress on stderr, so check exit codes instead of using $ErrorActionPreference = "Stop".
function Run([scriptblock]$cmd) {
  & $cmd
  if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $cmd" }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  throw "uv is not installed. Install it with: powershell -ExecutionPolicy ByPass -c ""irm https://astral.sh/uv/install.ps1 | iex"""
}
if ($Recreate -and (Test-Path .venv)) { Remove-Item .venv -Recurse -Force }
if (-not (Test-Path .venv)) { Run { uv venv .venv --python 3.13 } }
$py = ".\.venv\Scripts\python.exe"

# torch/torchvision come from the PyTorch index; everything else they need falls back to PyPI.
if ($Cpu) { $torchIndex = "https://download.pytorch.org/whl/cpu" }
else      { $torchIndex = "https://download.pytorch.org/whl/cu126" }   # CUDA 12.6 build; RTX 2080 (driver >= 560 recommended)
Run { uv pip install --python $py torch==2.7.1 torchvision==0.22.1 --index-url $torchIndex --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match }
Run { uv pip install --python $py -r requirements-dev.txt }

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env - edit DATA_ROOT in it." }

Push-Location frontend
Run { npm install --no-audit --no-fund }
Pop-Location

Run { & $py -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" }
