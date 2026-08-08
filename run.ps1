param(
    [int]$Port = 8000,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    python -m venv (Join-Path $ProjectRoot ".venv")
}

if (-not $SkipInstall) {
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}

& $VenvPython (Join-Path $ProjectRoot "scripts\download_models.py")
Set-Location -LiteralPath $ProjectRoot
& $VenvPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port

