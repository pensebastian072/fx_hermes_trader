# Add GPU research extras (torch cu121 + huggingface_hub) to the existing
# fx_hermes_trader venv, for gpu_meanrev/. ASCII-only on purpose (PowerShell
# 5.1 mangles non-ASCII). Run from the repo root. Idempotent.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    throw "venv not found at $py -- create it first (see CLAUDE.md)"
}

# torch from the cu121 index; --native-tls for this box's TLS interception.
uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cu121 --native-tls
uv pip install --python $py -r (Join-Path $repo "requirements-gpu.txt") --native-tls

& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
& $py -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
