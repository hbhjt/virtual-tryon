param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$aiPython = Join-Path $projectRoot ".venv-ai\Scripts\python.exe"
$vendorRoot = Join-Path $projectRoot "vendor"
$catvtonRoot = Join-Path $vendorRoot "CatVTON"

if (-not (Test-Path $aiPython)) {
    python -m venv (Join-Path $projectRoot ".venv-ai")
}

if (-not $SkipInstall) {
    & $aiPython -m pip install --upgrade pip
    & $aiPython -m pip install "torch==2.8.0" torchvision --index-url https://download.pytorch.org/whl/cpu
    & $aiPython -m pip install "diffusers==0.29.2" "transformers==4.53.3" "accelerate>=0.31" huggingface_hub pillow numpy scipy safetensors tqdm opencv-python-headless
}

New-Item -ItemType Directory -Force -Path $vendorRoot | Out-Null
if (-not (Test-Path (Join-Path $catvtonRoot ".git"))) {
    git clone --filter=blob:none https://github.com/Zheng-Chong/CatVTON.git $catvtonRoot
}
git -C $catvtonRoot checkout 3b795364a4d2f3b5adb365f39cdea376d20bc53c

Write-Host "AI 高质量环境已就绪。首次在网页选择 AI 高质量时会下载约 5GB 模型。"
